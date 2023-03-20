


def price(model, input_tokens, output_tokens):
    # https://openai.com/pricing
    if model in ['gpt-3.5-turbo', 'gpt-3.5']:
        tokens = input_tokens + output_tokens
        dollars = 0.002 * tokens / 1000
    elif model == 'gpt-4':
        dollars =  (.03*input_tokens + .06*output_tokens) / 1000
    else:
        raise ValueError(f"model {model} not supported")
    return dollars
