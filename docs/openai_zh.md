# 兼容 OpenAI 接口
[Tigerbot API](https://www.tigerbot.com/api-reference/chat) 已经兼容 [OpenAI API](https://platform.openai.com/docs/introduction) 部分功能。您可以通过 [OpenAI Python Library](https://github.com/openai/openai-python) 使用我们的服务。目前已经兼容以下功能：
* [Chat Completions](#chat-completion-功能)


## 安装
需要预先安装 [OpenAI Python Library](https://github.com/openai/openai-python)。可以通过以下命令安装：
```shell
pip install --upgrade openai
```

## Chat Completions 功能
[Tigerbot API](https://www.tigerbot.com/api-reference/chat) 已经兼容 [OpenAI API](https://platform.openai.com/docs/introduction) 的 Chat completion 功能。


### 兼容性
我们正在努力与 OpenAI 的接口进行兼容，但当前仍有一些不同点需要注意：
* [Tigerbot API](https://www.tigerbot.com/api-reference/chat) 的 Chat Completions 支持互联网搜索功能，此功能可以通过 OpenAI 的库进行使用，只需传入额外的 `internet` 参数：
```python
import openai

openai.api_base = "http://api.tigerbot.com/openai"
openai.api_key = '<YOUR_TIGERBOT_API_KEY>'

response = openai.ChatCompletion.create(
    model='tigerbot-7b-sft',
    messages=[
        {
            'role': 'user',
            'content': 'aapl'
        }
    ],
    internet=True
)

print(response)
```
* 支持的 OpenAI 接口参数（参数定义参见 [OpenAI Python Library](https://github.com/openai/openai-python)）：
  * `max_tokens`
  * `temperature`
  * `top_p`
  * `stream`
  * `model`
* 额外参数：
  * `internet`: `internet=True` 开启互联网搜索模式，默认关闭。
* `finish_reason` 为 `sensitive` 是表示输入或者输出中有敏感词，此时 `content` 将返回空。

### 示例

以下示例是根据 [OpenAI Cookbook](https://github.com/openai/openai-cookbook) 中的示例修改而来。字段具体含义及用法可以参考 [OpenAI Python Library](https://github.com/openai/openai-python) 及 [OpenAI Cookbook](https://github.com/openai/openai-cookbook)。

#### Chat Completions

运行以下 Python 代码
```python
import openai

openai.api_base = "http://api.tigerbot.com/openai"
openai.api_key = '<YOUR_TIGERBOT_API_KEY>'

response = openai.ChatCompletion.create(
    model='tigerbot-7b-sft',
    messages=[
        {
            'role': 'user',
            'content': 'Count to 10.'
        }
    ]
)

print(f"Full response received:\n{response}")

reply = response['choices'][0]['message']
print(f"Extracted reply: \n{reply}")

reply_content = response['choices'][0]['message']['content']
print(f"Extracted content: \n{reply_content}")

```
运行完毕后，将打印以下内容：
```shell
Full response received:
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "One, two, three, four, five, six, seven, eight, nine, ten."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 21,
    "total_tokens": 27
  },
  "id": "chatcmpl-1c336809e0ee4384bb57128ccd7ca583",
  "object": "chat.completion",
  "created": 1691046054,
  "model": "tigerbot-7b-sft"
}
Extracted reply:
{
  "role": "assistant",
  "content": "One, two, three, four, five, six, seven, eight, nine, ten."
}
Extracted content:
One, two, three, four, five, six, seven, eight, nine, ten.
```

#### 流式 Chat completion
运行以下 Python 代码
```python
import openai

openai.api_base = "http://api.tigerbot.com/openai"
openai.api_key = '<YOUR_TIGERBOT_API_KEY>'

response = openai.ChatCompletion.create(
    model='tigerbot-7b-sft',
    messages=[
        {'role': 'user', 'content': "Count to 10."}
    ],
    stream=True
)

for chunk in response:
    print(chunk)

```
运行完毕后，将打印以下内容：
<details>
  <summary>
    查看打印内容
  </summary>

```shell
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "One, "
      }
    }
  ],
  "id": "chatcmpl-f23607a17a074f7c944264cfa8eeadb4",
  "object": "chat.completion",
  "created": 1691046229,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "two, "
      }
    }
  ],
  "id": "chatcmpl-6f717811f64b4c6ba28850b380af5d34",
  "object": "chat.completion",
  "created": 1691046229,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "three, "
      }
    }
  ],
  "id": "chatcmpl-85c4b4d2da0f43fb95774a39eeab0a5f",
  "object": "chat.completion",
  "created": 1691046229,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "four, "
      }
    }
  ],
  "id": "chatcmpl-7132d03a60704199a0acbeeccc2f4b34",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "five, "
      }
    }
  ],
  "id": "chatcmpl-74de03e2b747475d8834c8c0a774de43",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "six, "
      }
    }
  ],
  "id": "chatcmpl-1df8660259ea47dfbf1553e29bfa45db",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "seven, "
      }
    }
  ],
  "id": "chatcmpl-018dd25c33bc47ec930e926817cfe0da",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "eight, "
      }
    }
  ],
  "id": "chatcmpl-bb416d621a5b4f4d854ca54d51b63e02",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "nine, "
      }
    }
  ],
  "id": "chatcmpl-1ad2ad68a1db4ae2a3bcc56775866d7c",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "ten."
      }
    }
  ],
  "id": "chatcmpl-48361ba7cf6f41c981de0cf069a1340a",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}
{
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "stop"
    }
  ],
  "id": "chatcmpl-0838fbfe755f4dbc852fead75a415acc",
  "object": "chat.completion",
  "created": 1691046230,
  "model": "tigerbot-7b-sft"
}

```

</details>
