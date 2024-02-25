# ToxStop Bot
ToxStop is an innovative bot designed for the BlueSky platform, aimed at identifying and mitigating toxic conversations in real-time. By leveraging advanced natural language processing (NLP) techniques, ToxStop provides a safer, more respectful online environment. When a user prompts ToxStop with @toxstop help or @toxstop stop tox, the bot activates its dual-function mechanism: first, detecting potential toxicity in the conversation, and then, generating a diffuser comment designed to de-escalate the situation.

## Features
- Toxicity Detection: Utilizes a pretrained model to accurately identify toxic content in conversations.
- Diffuser Comment Generation: Employs a language model (LLM) like GPT to create responses that aim to de-escalate tensions.
- Easy Activation: Users can summon ToxStop with simple commands, making it accessible to everyone in the conversation.
# Getting Started

## Prerequisites
BlueSky account and API access
Python 3.8 or later
Access to a pretrained toxicity classifier
Access to a language model capable of generating text (e.g., GPT)

## Installation
Clone the ToxStop repository:
```
git clone 
cd toxstop
```
Install required Python packages:
```
pip install -r requirements.txt
```
Configure your .env file with necessary API keys and model paths.
Usage
Deploy ToxStop to a server with BlueSky API access.
Ensure ToxStop is registered and active on your BlueSky account.
To use ToxStop, simply type @toxstop help or @toxstop stop tox in any conversation.
How It Works
Below is a simplified processing diagram illustrating ToxStop's operation:

```
+-----------------+        +-----------------------+        +-------------------+
|                 |        |                       |        |                   |
|  User Prompt    +------->+  Toxicity Detection   +------->+  Diffuser Comment |
|  (@toxstop ...) |        |  (Pretrained Model)   |        |  Generation (LLM) |
|                 |        |                       |        |                   |
+-----------------+        +-----------------------+        +-------------------+
                                       |                              |
                                       |                              |
                                       v                              v
                              +------------------+           +---------------------+
                              |                  |           |                     |
                              |  BlueSky API     |           |  Generate Response  |
                              |  (Aproto)        |           |  to Prevent         |
                              |                  |           |  Conversation       |
                              |                  |           |  Continuation       |
                              +------------------+           +---------------------+
```
Contributions

Contributions are welcome! Please submit a pull request or open an issue if you have suggestions for improvements.

License
Specify your license here (e.g., MIT, GPL, etc.)


For generating non-toxic and contextually appropriate responses, especially when aiming to mitigate conversation toxicity or detect toxic comments, setting the right parameters is crucial for balancing creativity with relevance and safety. Based on the requirements you've outlined (low temperature and max tokens around 100), here are the suggested parameters:

```
input={
    "top_k": 40,  # Controls diversity. A lower value than 50 to slightly reduce randomness.
    "top_p": 0.9,  # Nucleus sampling. Lowering a bit from 0.95 to make responses slightly less varied.
    "prompt": your_prompt_here,  # Replace with your actual prompt.
    "temperature": 0.5,  # Lower temperature for more predictable responses.
    "max_new_tokens": 100,  # Limiting the response length as per your request.
    "min_new_tokens": 50,  # Ensuring a minimum length to provide enough content for a meaningful response.
    "repetition_penalty": 1.2  # Slightly higher to discourage repetitive responses.
}
```
top_k: Reducing top_k from 50 to 40 makes the model's choices slightly less random, focusing more on the likely words while still allowing for some creativity. This is useful for ensuring that the generated content remains relevant and coherent.

top_p (nucleus sampling): Adjusting top_p to 0.9 from 0.95 tightens the range of words considered for each choice, further concentrating on the more probable outcomes. It's a balance between generating diverse content and avoiding too much randomness, which is important for sensitive applications like mitigating toxicity.

temperature: Setting the temperature to 0.5. A lower temperature makes the model's responses more deterministic and conservative, which is suitable for generating responses where accuracy and relevance are more critical than creativity, such as in diffusing toxic situations.

max_new_tokens and min_new_tokens: Specifying a maximum of 100 new tokens and a minimum of 50 ensures that the responses are concise yet sufficiently detailed to convey a complete thought or argument, aligning with the goal of providing meaningful engagement without overwhelming the conversation.

repetition_penalty: Increasing the repetition penalty slightly to 1.2 encourages the model to use a wider range of vocabulary and sentence structures, reducing the risk of generating monotonous or repetitive responses.

These parameters are tailored to generate responses that are coherent, contextually relevant, and conservative in terms of creativity, which should be effective for applications aiming at detecting toxicity and generating diffusing responses in online conversations. Adjustments might be needed based on the specific behavior of the model you're using and the exact nature of the conversations you're dealing with.