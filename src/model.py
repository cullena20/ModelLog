"""
Main model wrapper class. 
Whenever we use models, it will be wrapped in this.
Loads model, sets up tokenizer, has essential methods for ModelLog evaluation
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pickle

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"

class ModelWrapper:
    def __init__(self, model_name, instruction_tuned=None):
        # Load model, tokenizer, and chat template based on model_name
        self.model_name = model_name
        self.model =  AutoModelForCausalLM.from_pretrained(
                            model_name,
                            dtype=torch.bfloat16,
                            device_map="auto"
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # IMPORTANT: without this, last_token_prob makes no sense
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = getattr(self.model.config, 'vocab_size', len(self.tokenizer))

        self.device = next(self.model.parameters()).device

        # Chat template support (used for instruction-tuned models).
        # Auto-detected from tokenizer; can be forced True by passing instruction_tuned=True,
        # in which case a default template is installed if the tokenizer lacks one.
        tokenizer_has_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None
        )
        if instruction_tuned is True and not tokenizer_has_template:
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        self.instruction_tuned = tokenizer_has_template if instruction_tuned is None else bool(instruction_tuned)
        self.has_chat_template = self.instruction_tuned  # alias for clarity in last_token_prob
        self.chat_suffix_len = self._compute_chat_suffix_len() if self.has_chat_template else 0
        self.supports_system_role = self._check_system_role_support() if self.has_chat_template else False

    def _compute_chat_suffix_len(self):
        """How many tokens the template appends after user content (no generation prompt)."""
        try:
            marker = "elephant"
            marker_ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
            template_ids = list(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}],
                add_generation_prompt=False,
                tokenize=True,
            ))
            n = len(marker_ids)
            for i in range(len(template_ids) - n, -1, -1):
                if template_ids[i:i + n] == marker_ids:
                    return len(template_ids) - (i + n)
            return 0
        except Exception:
            return 0

    def _check_system_role_support(self):
        """Check if the tokenizer's chat template accepts a system role."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
                add_generation_prompt=False,
                tokenize=False,
            )
            return True
        except Exception:
            return False

    def last_token_prob(self, dataloader, desc_prefix="", use_chat_template=False, yes_no=False):
        """
        Computes P(last_token | previous_tokens).

        Modes:
          use_chat_template=False (default): plain tokenization; pred position is -2 (predicts answer token at -1).
          use_chat_template=True, yes_no=False: chat template without generation prompt; finds answer token just before closing template tokens.
          use_chat_template=True, yes_no=True: chat template with generation prompt; pred position is -1 (next-token distribution for Yes/No calibration).

        Returns:
            List[dict] aligned with dataset order.
        """

        all_stats = []

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{desc_prefix}Computing last-token probs"):
                if not use_chat_template:
                    # Non-instruction-tuned: plain tokenization, answer token is at position -1
                    tokens = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    pred_position = -2
                    answer_position = -1
                elif yes_no:
                    # Yes/no eval: chat template with generation prompt, predict next token.
                    # apply_chat_template with return_tensors returns a raw tensor (no attention_mask),
                    # so format to strings first, then tokenize properly.
                    if self.supports_system_role:
                        messages = [
                            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": t}]
                            for t in batch
                        ]
                    else:
                        messages = [[{"role": "user", "content": t}] for t in batch]
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    tokens = self.tokenizer(
                        formatted,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    pred_position = -1
                    answer_position = -1  # not meaningful; calibrate_true_false_dataset uses full_prob_vector
                else:
                    # Instruction-tuned standard eval: chat template without generation prompt.
                    # The answer token sits just before the closing template tokens.
                    messages = [[{"role": "user", "content": t}] for t in batch]
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=False,
                    )
                    tokens = self.tokenizer(
                        formatted,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    # chat_suffix_len closing tokens follow the answer token
                    pred_position = -(self.chat_suffix_len + 2)
                    answer_position = -(self.chat_suffix_len + 1)

                x_ids = tokens["input_ids"].to(self.model.device)
                attn = tokens["attention_mask"].to(self.model.device)

                outputs = self.model(input_ids=x_ids, attention_mask=attn)
                logits = outputs.logits  # (batch, seq_len, vocab_size)

                batch_size = x_ids.size(0)

                for b in range(batch_size):
                    last_token_logits = logits[b, pred_position, :].to(torch.float32)
                    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
                    probs = torch.exp(log_probs)

                    last_token_id = x_ids[b, answer_position].item()

                    all_stats.append({
                        "last_token": self.tokenizer.decode([last_token_id]),
                        "last_token_id": last_token_id,
                        "logprob": log_probs[last_token_id].item(),
                        "probability": probs[last_token_id].item(),
                        "full_logprob_vector": log_probs.detach().cpu(),
                        "full_prob_vector": probs.detach().cpu(),
                    })


        return all_stats