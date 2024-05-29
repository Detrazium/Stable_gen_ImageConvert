from diffusion_stable import stt
url = "https://s565vla.storage.yandex.net/rdisk/2b77dd8e29831c1392ed9d2ac5a03b14b886d99b8892ccf91d5e9d426ec6d9aa/6654d918/z2Hg9rf1o6KeuJ7Nd_tI11Ucr2_IGd7VQopCEeOZheReHV5r3AaW2bzTJxdROH5vu-w9jIL9uEQ6UQ-U-o4Z4A==?uid=0&filename=Image_.jpg&disposition=inline&hash=&limit=0&content_type=image%2Fjpeg&owner_uid=0&fsize=191565&hid=7d1fdb13425cb90c9a42494da164a660&media_type=image&tknv=v2&etag=abd8ea2d21daebe70dd23d95055c5d6b&ts=619742d967600&s=79ad919136cf8243ef25ed7a27e1daf832e156e4799e7e4cfbfff8befa8c000b&pb=U2FsdGVkX1_bW3Q62dbd-08TZSpNeNF7X3YKpkmOWFLCPqcYZE2RFjePfqgcSKJjSi1XPCJd8YeKpM1vKacnd-PRJN02kKJxu1QJaeLCfFc"

prompt = "masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd"

def main():
    se = stt().start(image=url, prompt=prompt)

if __name__ == "__main__":
    main()