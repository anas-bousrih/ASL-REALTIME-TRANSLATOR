from app.output.llm_rewrite import LLMRewriter


TEST_CASES = [
    ["I", "GO", "SCHOOL"],
    ["YOU", "NAME", "WHAT"],
    ["YESTERDAY", "PARTY", "FUN"],
    ["ME", "WANT", "WATER"],
    ["TODAY", "WEATHER", "GOOD"],
]


def main():
    rewriter = LLMRewriter(
        enabled=True,
        model="llama3.2:3b",
        interval_sec=0.0,
        min_tokens=1,
    )

    for tokens in TEST_CASES:
        text = " ".join(tokens)
        out = rewriter.rewrite_now(text)
        print("-" * 60)
        print("INPUT :", text)
        print("OUTPUT:", out)


if __name__ == "__main__":
    main()
