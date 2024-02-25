from run_model import run, predictor


def main():
    conversations = [
        [
            "Hey, how are you?",
            "I'm doing well, thanks for asking. How about you?",
            "I'm doing great, thanks!",
        ],
        [
            "Hey, how are you?",
            "What's it to you?",
        ],
    ]
    print(run(conversations, predictor))


if __name__ == "__main__":
    main()