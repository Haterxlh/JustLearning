from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "用一句话说明 KV Cache 的作用"}],
    max_tokens=128,
)

print(response.choices[0].message.content)
print(response.usage)