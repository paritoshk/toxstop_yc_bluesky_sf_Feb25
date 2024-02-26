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
from run_model import run, predictor
# from dotenv import load_dotenv
# from openai import OpenAI
import replicate
# Functions that generate diffuser prompt 
# import replicate
# Load environment variables
# load_dotenv()
BLUESKY_USERNAME = "paritoshk.bsky.social"
BLUESKY_PASSWORD = "Pari2798!"
# Create a Bluesky client
client = Client("https://bsky.social")
firehose = FirehoseSubscribeReposClient()

 #can you try?
# REPLICATE_API_TOKEN = 'r8_7ICSWEeEQdH6SyLPpcjU1723lKbVExK1kqyIx'
# Example function

# Assuming REPLICATE_API_TOKEN and other necessary imports are defined elsewhere
from openai import OpenAI
# OPENAI_KEY= "sk-f0uYfeqTyHguaAHJSSy4T3BlbkFJcc7wucomments[:-1p]l6MpzlL7w1alZYG"
OPENAI_KEY="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"
openaiclient = OpenAI(api_key=OPENAI_KEY)

def detect_toxic_comments(thread_info: str) -> list:
    """
    Analyze the thread information to detect toxic comments using OpenAI.
    """
    detection_prompt = f"Identify any toxic comments from the following thread: {thread_info}"
    completion = openaiclient.chat.completions.create(
        model="gpt-4-0125-preview",  # Adjust the model as necessary
        messages=[
            {"role": "system", "content": "You are an AI trained to identify toxic behavior in text."},
            {"role": "user", "content": detection_prompt}
        ],
        temperature=0.15,
        max_tokens=80,
    )
    
    # Assuming the model returns a list of toxic comments directly
    toxic_comments = completion.choices[0].message.content.split("\n")
    return toxic_comments

def generate_diffuser_text(thread_info: str) -> str:
    """
    Generate a response aimed at diffusing the toxicity identified in the thread using OpenAI.
    """
    toxic_summary = " ".join(thread_info)
    diffuser_prompt = f"With the context of this thread: {thread_info}, generate a short response that can help diffuse the situation."
    
    completion = openaiclient.chat.completions.create(
        model="gpt-4-0125-preview",  # Adjust the model as necessary
        messages=[
            {"role": "system", "content": "You are an AI trained to promote positive interactions and mitigate toxicity."},
            {"role": "user", "content": diffuser_prompt}
        ],
        temperature=0.15,
        max_tokens=80,
    )
    
    diffuser_text = completion.choices[0].message.content
    return diffuser_text



def get_thread_text(record):
    root_uri = record["reply"]["root"]["uri"]
    posts = client.get_post_thread(uri=root_uri)
    posts = posts.model_dump()
    record_uri = record["uri"]

    def _dfs(tree):
        text = tree["post"]["record"]["text"]
        uri = tree["post"]["uri"]
        if tree["replies"] is None or len(tree["replies"]) == 0:
            if uri == record_uri:
                return [text]
            else:
                return None

        for reply in tree["replies"]:
            path = _dfs(reply)
            if path is not None:
                return [text] + path

    return _dfs(posts["thread"])
    



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
                print("HIII")
                # Compile thread information and detect toxic comments.
                if "reply" not in record:
                    return
                # posts_in_thread = client.get_post_thread(uri=record["reply"]["root"]["uri"])
                post_texts = get_thread_text(record)
                print(post_texts)
                predictions, scores = run([post_texts[:-1]], predictor)
                print(predictions, scores)
                if predictions.item() == 0:
                    return
                
                # Generate a diffuser text based on the thread and toxic comments.
                diffuser_text = generate_diffuser_text(post_texts)
                # diffuser_text = "Let's keep our conversations respectful and constructive. Positive communication builds a better community."
                print(diffuser_text)
                # Get the poster's profile for personalization.
                poster_profile = client.get_profile(actor=record["author"])
                
                # Send a personalized reply to the post using the generated diffuser text.
                parent_ref = {"uri": record["reply"]["parent"]["uri"], "cid": record["reply"]["parent"]["cid"]}
                root_ref = {"uri": record["reply"]["root"]["uri"], "cid": record["reply"]["root"]["cid"]}
                reply_ref = models.AppBskyFeedPost.ReplyRef(parent=parent_ref, root=root_ref)
                client.send_post(
                    reply_to=reply_ref,
                    text=diffuser_text.strip('"')[:280],
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