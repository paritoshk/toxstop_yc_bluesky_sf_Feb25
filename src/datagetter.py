import os
from atproto import (
    CAR,
    AtUri,
    Client,
    FirehoseSubscribeReposClient,
    firehose_models,
    models,
    parse_subscribe_repos_message,
)
from dotenv import load_dotenv
from openai import OpenAI
import replicate
# Functions that generate diffuser prompt 
# import replicate
# Load environment variables
load_dotenv()
BLUESKY_USERNAME = "paritoshk.bsky.social"
BLUESKY_PASSWORD = "Pari2798!"
# Create a Bluesky client
client = Client("https://bsky.social")
firehose = FirehoseSubscribeReposClient()

 #can you try?
#REPLICATE_API_TOKEN = 'r8_7ICSWEeEQdH6SyLPpcjU1723lKbVExK1kqyIx'
# Example function

# Assuming REPLICATE_API_TOKEN and other necessary imports are defined elsewhere

def detect_toxic_comments(thread_info: str) -> list:
    """
    Analyze the thread information to detect toxic comments using Replicate.

    :param thread_info: A string containing the compiled content of the thread.
    :return: A list of strings, each a comment identified as toxic.
    """
    detection_prompt = f"Identify any toxic comments from the following thread: {thread_info}"
    #os.environ['REPLICATE_API_TOKEN'] = 'r8_7ICSWEeEQdH6SyLPpcjU1723lKbVExK1kqyIx'
    # Replace 'model_name' with your actual model name or ID on Replicate
    output = replicate.run(
        "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
        input={
            "prompt": detection_prompt,
            "top_k": 40,  # Controls diversity. A lower value than 50 to slightly reduce randomness.
            "top_p": 0.9,  # Nucleus sampling. Lowering a bit from 0.95 to make responses slightly less varied.
            "temperature": 0.5,  # Lower temperature for more predictable responses.
            "max_new_tokens": 100,  # Limiting the response length as per your request.
            "min_new_tokens": 50,  # Ensuring a minimum length to provide enough content for a meaningful response.
            "repetition_penalty": 1.2  # Slightly higher to discourage repetitive responses.
            # Add other parameters as required by your model
        }
    )
    # Example parsing, adjust based on actual output format from your model
    #toxic_comments = output[0]["text"].split("\n")  # Adjust this line based on the model's output
    return output

def generate_diffuser_text(thread_info: str, toxic_comments: list) -> str:
    """
    Generate a response aimed at diffusing the toxicity identified in the thread using Replicate.

    :param thread_info: A string containing the compiled content of the thread.
    :param toxic_comments: A list of strings, each a comment identified as toxic.
    :return: A string containing the AI-generated diffuser text.
    """
    toxic_summary = " ".join(toxic_comments)  # Combine toxic comments into a single string for simplicity
    diffuser_prompt = f"With the context of this thread: {thread_info} and the following toxic comments: {toxic_summary}, generate a response that can help diffuse the situation."
    # Replace 'model_name' with your actual model name or ID on Replicate
    output = replicate.run(
        "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
        input={
            "prompt": diffuser_prompt,
            "top_k": 40,  # Controls diversity. A lower value than 50 to slightly reduce randomness.
            "top_p": 0.9,  # Nucleus sampling. Lowering a bit from 0.95 to make responses slightly less varied.
            "temperature": 0.5,  # Lower temperature for more predictable responses.
            "max_new_tokens": 100,  # Limiting the response length as per your request.
            "min_new_tokens": 50,  # Ensuring a minimum length to provide enough content for a meaningful response.
            "repetition_penalty": 1.2  # Slightly higher to discourage repetitive responses.
            # Add other parameters as required by your model
        }
    )
    import pdb; pdb.set_trace()
    #diffuser_text = output[0]["text"]  # Adjust this line based on the model's output
    return output

def process_operation(
    op: models.ComAtprotoSyncSubscribeRepos.RepoOp,
    car: CAR,
    commit: models.ComAtprotoSyncSubscribeRepos.Commit,
) -> None:
    uri = AtUri.from_str(f"at://{commit.repo}/{op.path}")
    if op.action == "create":
        if not op.cid:
            return
        record = car.blocks.get(op.cid)
        if not record:
            return
        record = {
            "uri": str(uri),
            "cid": str(op.cid),
            "author": commit.repo,
            **record,
        }
        if uri.collection == models.ids.AppBskyFeedPost:
            # Check for the intervention trigger in the post's text.
            if "stop-tox" in record["text"] or "@stoptox" in record["text"]:
                # Compile thread information and detect toxic comments.
                posts_in_thread = client.get_post_thread(uri=record["uri"])
                thread_info = str(posts_in_thread)
                # Adjust the index based on actual data structure
                toxic_comments = detect_toxic_comments(thread_info)
                
                # Generate a diffuser text based on the thread and toxic comments.
                if toxic_comments:
                    diffuser_text = generate_diffuser_text(thread_info, toxic_comments)
                else:
                    diffuser_text = "Let's keep our conversations respectful and constructive. Positive communication builds a better community."
                print(diffuser_text)
                # Get the poster's profile for personalization.
                poster_profile = client.get_profile(actor=record["author"])
                
                # Send a personalized reply to the post using the generated diffuser text.
                record_ref = {"uri": record["uri"], "cid": record["cid"]}
                reply_ref = models.AppBskyFeedPost.ReplyRef(parent=record_ref, root=record_ref)
                client.send_post(
                    reply_to=reply_ref,
                    text=f"Hey, {poster_profile.display_name}. {diffuser_text}",
                )
    elif op.action == "delete":
        # Process delete(s)
        return
    elif op.action == "update":
        # Process update(s)
        return
    return



# No need to edit this function - it processes messages from the firehose
def on_message_handler(message: firehose_models.MessageFrame) -> None:
    commit = parse_subscribe_repos_message(message)
    if not isinstance(
        commit, models.ComAtprotoSyncSubscribeRepos.Commit
    ) or not isinstance(commit.blocks, bytes):
        return
    car = CAR.from_bytes(commit.blocks)
    for op in commit.ops:
        process_operation(op, car, commit)


def main() -> None:
    client.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)
    print("🤖 Bot is listening")
    firehose.start(on_message_handler)


if __name__ == "__main__":
    main()