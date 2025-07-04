import torch
from unittest.mock import MagicMock

# Important: This test script assumes it is in the same directory as your sft_sarcasm_s1.py
# so it can import the necessary functions.
from sft_sarcasm_s1 import prepare_dataset, collate_fn, Qwen2VLProcessor


def create_mock_processor():
    """Creates a mock processor and tokenizer for testing purposes."""
    mock_tokenizer = MagicMock()
    
    # Let's define a simple vocabulary and how the tokenizer should behave
    vocab = {
        '<|im_start|>': 151644, 'system': 92542, '<|im_end|>': 151645,
        'user': 123, 'assistant': 456,
        'text:': 789, 'sarcasm_prompt': 1000,
        '<think>': 2000, 'reasoning': 2001, '</think>': 2002,
        '<bbox_2d>': 3000, 'none': 3001, '</bbox_2d>': 3002,
        '<answer>': 4000, 'sarcasm': 4001, '</answer>': 4002,
        '<PAD>': 0, '<IMAGE>': 151652,
    }
    
    # This function simulates the tokenizer's behavior for different strings
    def mock_tokenize_func(text_input, **kwargs):
        if isinstance(text_input, str):
            # Simplistic mapping for the test case
            if "You are a multimodal satire analysis assistant" in text_input: # System prompt
                return [vocab['<|im_start|>'], vocab['system']]
            if "Text:some text" in text_input: # User prompt
                return [vocab['<|im_start|>'], vocab['user'], vocab['text:'], vocab['sarcasm_prompt'], vocab['<IMAGE>'], vocab['<|im_end|>']]
            if "<think>The model provides its reasoning here.</think>" in text_input: # Assistant response
                return [
                    vocab['<|im_start|>'], vocab['assistant'],
                    vocab['<think>'], vocab['reasoning'], vocab['</think>'],
                    vocab['<bbox_2d>'], vocab['none'], vocab['</bbox_2d>'],
                    vocab['<answer>'], vocab['sarcasm'], vocab['</answer>'],
                    vocab['<|im_end|>']
                ]
            if "<answer>" in text_input and "<think>" not in text_input: # just the answer part
                 return [vocab['<answer>'], vocab['sarcasm'], vocab['</answer>']]
            if text_input == "<answer>":
                return [vocab['<answer>']]

        # For the real collate_fn, we need to handle the dict from apply_chat_template
        # This part is complex to mock perfectly, so we'll simplify and focus on the output
        # Let's mock the direct calls we know will happen
        if isinstance(text_input, list): # a batch of texts
             # In our test, we only pass one text
             full_text = text_input[0]
             # This simulates the full tokenized sequence
             return {'input_ids': torch.tensor([[
                151644, 92542, 102, 151645, # System prompt part
                151644, 123, 789, 1000, 151652, 151645, # User prompt part
                151644, 456, # Assistant start
                2000, 2001, 2002, # <think>...</think>
                3000, 3001, 3002, # <bbox_2d>...</bbox_2d>
                4000, 4001, 4002, # <answer>...</answer>
                151645 # Assistant end
             ]])}
        
        # for response_token_ids = processor.tokenizer(...)
        if isinstance(text_input, str) and "<think>" in text_input:
             return { 'input_ids': [
                2000, 2001, 2002, # <think>...</think>
                3000, 3001, 3002, # <bbox_2d>...</bbox_2d>
                4000, 4001, 4002, # <answer>...</answer>
             ]}

        return {'input_ids': [vocab.get(t, -1) for t in text_input.split()]}

    mock_tokenizer.side_effect = mock_tokenize_func
    mock_tokenizer.pad_token_id = vocab['<PAD>']
    mock_tokenizer.encode = lambda text, **kwargs: mock_tokenize_func(text)['input_ids']
    mock_tokenizer.convert_tokens_to_ids = lambda token: vocab.get(token, -1)
    
    # The main tokenizer call used by the collator
    def main_tokenizer_call(text, **kwargs):
        return mock_tokenize_func(text)
    
    mock_tokenizer.__call__ = main_tokenizer_call

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer
    # Mock apply_chat_template to just return the string, as in the real code
    mock_processor.apply_chat_template = lambda messages, **kwargs: "dummy string for collator"
    # Mock vision processing
    mock_processor.image_token = "<IMAGE>"
    
    # This is a global variable in sft_sarcasm_s1.py, we need to mock it
    global processor
    processor = mock_processor

    return mock_processor


def run_test():
    """Runs a test of the collate_fn."""
    print("--- Setting up Test ---")
    
    # Since collate_fn uses global `processor`, we must mock it globally.
    mock_proc = create_mock_processor()
    
    # We also need to patch the global IMG_DIR used by prepare_dataset
    import sft_sarcasm_s1
    sft_sarcasm_s1.IMG_DIR = "./" 
    # And the process_vision_info function
    sft_sarcasm_s1.process_vision_info = MagicMock(return_value=([], [], None))


    # 1. Create a sample data point
    example_data = {
        'image_id': 'test_id',
        'text': 'some text',
        'label': 1 # Sarcasm
    }
    print(f"Original data: {example_data}")

    # 2. Run it through prepare_dataset
    prepared_example = prepare_dataset(example_data)
    print("\nData after prepare_dataset is ready.")

    # 3. Run the prepared data through our collate_fn
    # The collate_fn expects a list of examples
    print("\n--- Running collate_fn ---")
    collated_batch = collate_fn([prepared_example])

    # 4. Analyze and print the results
    input_ids = collated_batch['input_ids'][0]
    labels = collated_batch['labels'][0]

    print(f"\nOriginal Input IDs:\n{input_ids.tolist()}")
    print(f"\nFinal Labels (with -100 for masked parts):\n{labels.tolist()}")

    # 5. Assertions to verify correctness
    print("\n--- Verifying Results ---")
    
    # Expected prompt length (system + user) is 12 tokens in our mock
    prompt_len = 12 
    # The response starts after the 'assistant' token, so at index 14
    response_start_index = 14
    
    # a) Assert that the prompt is masked
    assert torch.all(labels[:response_start_index] == -100).item(), "Test Failed: Prompt part is not fully masked!"
    print("✅ PASS: Prompt is correctly masked.")

    # b) Assert that the <think> and <bbox_2d> parts are masked
    # These are 6 tokens in our mock, from index 14 to 19
    think_bbox_part = labels[response_start_index : response_start_index + 6]
    assert torch.all(think_bbox_part == -100).item(), "Test Failed: <think>/<bbox_2d> part is not masked!"
    print("✅ PASS: <think> and <bbox_2d> parts are correctly masked.")

    # c) Assert that the <answer> part is NOT masked
    # This is 3 tokens from index 20 to 22
    answer_part = labels[response_start_index + 6 : response_start_index + 9]
    assert torch.all(answer_part != -100).item(), "Test Failed: <answer> part IS masked, but should not be!"
    print("✅ PASS: <answer> part is correctly preserved.")

    print("\n🎉 Test finished successfully! 🎉")


if __name__ == "__main__":
    run_test() 