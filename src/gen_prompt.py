# @title gen_prompt_with_gpt
import threading

from openai import OpenAI


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def gen_prompt_with_gpt(api_key: str, prompt: str, image_b64: str):
    client = OpenAI(api_key=api_key)
    system_prompt = f"""
    Generate a concise scene description (under 30 words in English) for AI drawing, focused on a product identified in an image, according to user requirements. Follow these steps:

    1. Identify the Product: Look at the image and succinctly describe the main product in simple terms (e.g., 'gym shoes', 'bottle').

    2. Create Scene Description Based on User Requirements: Using the identified product, develop a scene description that aligns with the user's specific request, ensuring it's different from the product's original setting. Format the description as '<Product> [in | on | close by | on top of | below | ...] [something], ...'.

    3. Adjust Scene Emphasis: Use '<important thing>+' to emphasize key elements and '<background thing>-' to de-emphasize others.

    Provide only the final scene description, tailored to the user's request.

    Example:
    Product Identified from Image: Chair
    User Requirement: A warm winter setting
    Possible Output: 'chair+ in a cozy, snow-view cabin-, with a crackling fireplace+.'

    User Requirement: ```{prompt}```
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_b64},
                },
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=2000,
        temperature=1.0
    )
    prompt = completion.choices[0].message.content
    print('  Image prompt: ' + prompt)
    return prompt


def gen_prompt(api_key: str, prompt: str, image_b64:str):
    return ThreadWithReturnValue(target=gen_prompt_with_gpt, args=(api_key, prompt, image_b64))
