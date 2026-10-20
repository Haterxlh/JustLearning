// src/api.test.ts —— 接口客户端的验收用例，代码写在 src/api.ts
//
// 约定：BASE_URL 在每次调用时读 import.meta.env.VITE_API_BASE，缺省 http://localhost:8000；
// 令牌存 localStorage 的 pj01_token；非 2xx 抛 ApiError，message 取后端的 detail。

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  clearToken,
  createConversation,
  getToken,
  listMessages,
  login,
  me,
  register,
  streamChat,
} from './api'

const BASE = 'http://api.test'
const TOKEN_KEY = 'pj01_token'

const user = {
  id: '11111111-1111-1111-1111-111111111111',
  email: 'a@example.com',
  created_at: '2026-10-20T12:00:00Z',
}

const conversation = {
  id: '22222222-2222-2222-2222-222222222222',
  title: '第一段对话',
  created_at: '2026-10-20T12:00:00Z',
  updated_at: '2026-10-20T12:00:00Z',
}

const message = {
  id: '33333333-3333-3333-3333-333333333333',
  role: 'user',
  content: '你好',
  created_at: '2026-10-20T12:00:00Z',
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function sseResponse(body: ReadableStream<Uint8Array>): Response {
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function sseStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk)
      controller.close()
    },
  })
}

function encode(text: string): Uint8Array {
  return new TextEncoder().encode(text)
}

function mockFetch(response: Response): ReturnType<typeof vi.fn> {
  // 每次调用都返回克隆，避免同一个 Response 的 body 被读第二次
  const fetchMock = vi.fn(async () => response.clone())
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function requestedUrl(fetchMock: ReturnType<typeof vi.fn>, index = 0): string {
  return String(fetchMock.mock.calls[index][0])
}

function requestedInit(fetchMock: ReturnType<typeof vi.fn>, index = 0): RequestInit {
  return fetchMock.mock.calls[index][1] as RequestInit
}

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE', BASE)
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('request 基础行为', () => {
  it('register 以 JSON 提交到 /api/auth/register 并返回用户', async () => {
    const fetchMock = mockFetch(jsonResponse(user, 201))

    const result = await register('a@example.com', '12345678')

    expect(requestedUrl(fetchMock)).toBe(`${BASE}/api/auth/register`)
    const init = requestedInit(fetchMock)
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ email: 'a@example.com', password: '12345678' })
    expect(result).toEqual(user)
  })

  it('非 2xx 抛 ApiError，status 与 detail 都带出来', async () => {
    mockFetch(jsonResponse({ detail: '邮箱已注册' }, 409))

    await expect(register('a@example.com', '12345678')).rejects.toMatchObject({
      status: 409,
      message: '邮箱已注册',
    })
    await expect(register('a@example.com', '12345678')).rejects.toBeInstanceOf(ApiError)
  })

  it('login 之后 getToken 拿到令牌，后续请求带 Authorization', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ access_token: 'tok-1', token_type: 'bearer', user }),
      )
      .mockResolvedValueOnce(jsonResponse(user))
    vi.stubGlobal('fetch', fetchMock)

    const auth = await login('a@example.com', '12345678')
    expect(auth.access_token).toBe('tok-1')
    expect(getToken()).toBe('tok-1')
    expect(localStorage.getItem(TOKEN_KEY)).toBe('tok-1')

    await me()
    const headers = new Headers(requestedInit(fetchMock, 1).headers)
    expect(headers.get('Authorization')).toBe('Bearer tok-1')
  })

  it('clearToken 之后请求不再带 Authorization', async () => {
    localStorage.setItem(TOKEN_KEY, 'tok-1')
    clearToken()
    expect(getToken()).toBeNull()

    const fetchMock = mockFetch(jsonResponse([]))
    await listMessages(conversation.id)
    const headers = new Headers(requestedInit(fetchMock).headers)
    expect(headers.get('Authorization')).toBeNull()
  })

  it('createConversation 不传标题时 body 是空对象', async () => {
    const fetchMock = mockFetch(jsonResponse(conversation, 201))

    await createConversation()

    expect(requestedUrl(fetchMock)).toBe(`${BASE}/api/conversations`)
    expect(JSON.parse(String(requestedInit(fetchMock).body))).toEqual({})
  })

  it('listMessages 拼出带会话 id 的路径并返回消息数组', async () => {
    const fetchMock = mockFetch(jsonResponse([message]))

    const result = await listMessages(conversation.id)

    expect(requestedUrl(fetchMock)).toBe(`${BASE}/api/conversations/${conversation.id}/messages`)
    expect(result).toEqual([message])
  })
})

describe('streamChat', () => {
  it('跨帧切块也能拼回两段增量，最后调用一次 onDone', async () => {
    const fetchMock = mockFetch(
      sseResponse(
        sseStream([
          encode('data: {"delta": "你'),
          encode('好"}\n\ndata: {"delta": "呀"}\n\ndata: [DONE]\n\n'),
        ]),
      ),
    )
    const onDelta = vi.fn()
    const onDone = vi.fn()
    const onError = vi.fn()

    await streamChat(conversation.id, '你好', { onDelta, onDone, onError })

    expect(onDelta.mock.calls.map((call) => call[0])).toEqual(['你好', '呀'])
    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onError).not.toHaveBeenCalled()
    expect(requestedUrl(fetchMock)).toBe(`${BASE}/api/chat/stream`)
    expect(JSON.parse(String(requestedInit(fetchMock).body))).toEqual({
      conversation_id: conversation.id,
      content: '你好',
    })
  })

  it('一个汉字的字节被切开时仍能正确解码', async () => {
    const full = encode('data: {"delta": "你"}\n\ndata: [DONE]\n\n')
    const charBytes = encode('你')
    const at = full.findIndex(
      (byte, index) =>
        byte === charBytes[0] &&
        full[index + 1] === charBytes[1] &&
        full[index + 2] === charBytes[2],
    )
    const onDelta = vi.fn()
    mockFetch(sseResponse(sseStream([full.slice(0, at + 1), full.slice(at + 1)])))

    await streamChat(conversation.id, '你好', { onDelta, onDone: vi.fn(), onError: vi.fn() })

    expect(onDelta).toHaveBeenCalledWith('你')
  })

  it('error 帧交给 onError，不调用 onDone', async () => {
    mockFetch(
      sseResponse(
        sseStream([encode('data: {"error": "模型接口调用失败：401"}\n\n')]),
      ),
    )
    const onDone = vi.fn()
    const onError = vi.fn()

    await streamChat(conversation.id, '你好', { onDelta: vi.fn(), onDone, onError })

    expect(onError).toHaveBeenCalledWith('模型接口调用失败：401')
    expect(onDone).not.toHaveBeenCalled()
  })

  it('会话不存在时抛 404 的 ApiError', async () => {
    mockFetch(jsonResponse({ detail: '会话不存在' }, 404))

    await expect(
      streamChat(conversation.id, '你好', { onDelta: vi.fn(), onDone: vi.fn(), onError: vi.fn() }),
    ).rejects.toMatchObject({ status: 404, message: '会话不存在' })
  })
})
