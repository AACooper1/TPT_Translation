from transformers import MT5ForConditionalGeneration, AutoConfig
from accelerate import Accelerator
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

accelerator = Accelerator()
model = MT5ForConditionalGeneration(AutoConfig.from_pretrained('google/mt5-base'))
model = accelerator.unwrap_model(model)

model = load_state_dict_from_zero_checkpoint(model, 'final_results/tpt_model')

model.push_to_hub('Asticky/latin_translation_tpt')

model = model.from_pretrained("Asticky/latin_translation_tpt")