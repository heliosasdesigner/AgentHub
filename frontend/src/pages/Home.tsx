import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send } from 'lucide-react'
import Mermaid from 'react-x-mermaid'
import '../App.css'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:11111'

interface Agent {
  id: string
  name: string
  description: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatHistory {
  [agentId: string]: Message[]
}

interface SessionState {
  [agentId: string]: string
}

interface FlowEntry {
  agent: string
  role: string
  message: string
}

interface SystemPrompt {
  id: string
  name: string
  content: string
  user_id?: string | null
  created_at?: string
  updated_at?: string
}

export function Home() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [activeAgents, setActiveAgents] = useState<string[]>([])
  const [systemPrompt, setSystemPrompt] = useState('')
  const [message, setMessage] = useState('')
  const [chatHistories, setChatHistories] = useState<ChatHistory>({})
  const [sessions, setSessions] = useState<SessionState>({})
  const [flowEntries, setFlowEntries] = useState<FlowEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [savedPrompts, setSavedPrompts] = useState<SystemPrompt[]>([])
  const [selectedPromptId, setSelectedPromptId] = useState<string>('custom')

  // Load agents on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/agents`)
      .then(res => res.json())
      .then(data => {
        setAgents(data)
        setActiveAgents(data.map((a: Agent) => a.id))
      })
      .catch(() => {
        // Fallback agents
        const fallbackAgents: Agent[] = [
          { id: 'rag-faiss', name: 'RAG-Faiss', description: 'RAG agent with Faiss vector store' },
          { id: 'graphrag-neo4j', name: 'GraphRAG-Neo4j', description: 'GraphRAG agent with Neo4j' },
          { id: 'cognee', name: 'Cognee', description: 'Cognee agent' },
          { id: 'leann', name: 'LEANN', description: 'Lightweight vector database with graph-based recomputation' },
        ]
        setAgents(fallbackAgents)
        setActiveAgents(fallbackAgents.map(a => a.id))
      })
  }, [])

  // Load saved system prompts on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/prompts`)
      .then(res => res.json())
      .then(data => setSavedPrompts(data))
      .catch(err => console.error('Failed to load prompts:', err))
  }, [])

  const getMermaidDiagram = (agentId: string) => {
    const diagrams: { [key: string]: string } = {
      'rag-faiss': `graph TD
  A[User Query] --> B[Retrieve from Faiss]
  B --> C[Vector Search]
  C --> D[Top K Documents]
  D --> E[LLM with Context]
  E --> F[Generated Response]`,
      'graphrag-neo4j': `graph TD
  A[User Query] --> B[Graph Query]
  B --> C[Neo4j Traversal]
  C --> D[Connected Entities]
  D --> E[LLM with Graph Context]
  E --> F[Generated Response]`,
      'cognee': `graph TD
  A[User Query] --> B[Cognee Pipeline]
  B --> C[Knowledge Processing]
  C --> D[Context Retrieval]
  D --> E[LLM Generation]
  E --> F[Generated Response]`,
      'leann': `graph TD
  A[User Query] --> B[Graph-based Index]
  B --> C[Selective Recomputation]
  C --> D[Semantic Search]
  D --> E[LLM with Context]
  E --> F[Generated Response]`,
    }
    return diagrams[agentId] || `graph TD
  A[User Query] --> B[Agent Processing]
  B --> C[Context Retrieval]
  C --> D[LLM Generation]
  D --> E[Generated Response]`
  }

  const handleAgentToggle = (agentId: string) => {
    setActiveAgents(prev =>
      prev.includes(agentId)
        ? prev.filter(id => id !== agentId)
        : [...prev, agentId]
    )
  }

  const handlePromptSelect = (value: string) => {
    setSelectedPromptId(value)
    if (value === 'custom') {
      setSystemPrompt('')
    } else {
      const prompt = savedPrompts.find(p => p.id === value)
      if (prompt) {
        setSystemPrompt(prompt.content)
      }
    }
  }

  const handleSubmit = async () => {
    if (!message.trim()) return
    if (activeAgents.length === 0) {
      alert('Please select at least one agent')
      return
    }

    setLoading(true)
    const userMessage = message
    setMessage('')

    const newHistories = { ...chatHistories }
    const newSessions = { ...sessions }
    const newFlows = [...flowEntries]

    for (const agentId of agents.map(a => a.id)) {
      if (!activeAgents.includes(agentId)) continue

      const history = newHistories[agentId] || []
      history.push({ role: 'user', content: userMessage })
      newHistories[agentId] = history

      try {
        const messages = []
        if (systemPrompt.trim()) {
          messages.push({ role: 'system', content: systemPrompt })
          newFlows.push({ agent: agentId, role: 'system', message: systemPrompt })
        }
        messages.push({ role: 'user', content: userMessage })

        const payload = {
          agent_id: agentId,
          session_id: newSessions[agentId] || undefined,
          messages,
          config: {},
        }

        const response = await fetch(`${BACKEND_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })

        const data = await response.json()
        const assistantReply = data.reply.content

        history.push({ role: 'assistant', content: assistantReply })
        newHistories[agentId] = history
        newSessions[agentId] = data.session_id || newSessions[agentId]

        newFlows.push({ agent: agentId, role: 'user', message: userMessage })
        newFlows.push({ agent: agentId, role: 'assistant', message: assistantReply })
      } catch (error) {
        const errorMsg = `Error from ${agentId}: ${error}`
        history.push({ role: 'assistant', content: errorMsg })
        newHistories[agentId] = history
        newFlows.push({ agent: agentId, role: 'user', message: userMessage })
        newFlows.push({ agent: agentId, role: 'assistant', message: errorMsg })
      }
    }

    setChatHistories(newHistories)
    setSessions(newSessions)
    setFlowEntries(newFlows)
    setLoading(false)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex flex-col p-6">
      {/* Title and subtitle */}
      <div className="mb-4">
        <h1 className="text-4xl font-bold mb-2">Multi-Agent Chat</h1>
        <p className="text-muted-foreground">Control which agents respond and share a system prompt across them.</p>
      </div>

      {/* Active agents selector */}
      <div className="mb-4">
        <Label className="text-base mb-2 block">Active agents</Label>
        <p className="text-sm text-muted-foreground mb-2">Toggle which agents should respond to each message.</p>
        <div className="flex flex-wrap gap-4">
          {agents.map(agent => (
            <div key={agent.id} className="flex items-center space-x-2">
              <Checkbox
                id={agent.id}
                checked={activeAgents.includes(agent.id)}
                onCheckedChange={() => handleAgentToggle(agent.id)}
              />
              <Label htmlFor={agent.id} className="cursor-pointer">{agent.id}</Label>
            </div>
          ))}
        </div>
      </div>

      {/* System prompt */}
      <div className="mb-4 space-y-3">
        <Label className="text-base block">System prompt (optional)</Label>
        <p className="text-sm text-muted-foreground">Select a saved prompt or type custom instructions.</p>

        <Select value={selectedPromptId} onValueChange={handlePromptSelect}>
          <SelectTrigger>
            <SelectValue placeholder="Select a saved prompt or type custom" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="custom">Custom (type below)</SelectItem>
            {savedPrompts.map(prompt => (
              <SelectItem key={prompt.id} value={prompt.id}>
                {prompt.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Textarea
          id="system-prompt"
          placeholder="Add instructions all agents should follow..."
          value={systemPrompt}
          onChange={(e) => {
            setSystemPrompt(e.target.value)
            if (selectedPromptId !== 'custom') {
              setSelectedPromptId('custom')
            }
          }}
          disabled={selectedPromptId !== 'custom' && selectedPromptId !== ''}
          rows={3}
          className={selectedPromptId !== 'custom' && selectedPromptId !== '' ? 'opacity-70' : ''}
        />
      </div>

      {/* Tabs section */}
      <Tabs defaultValue="chatroom" className="flex-1 mb-4">
        <TabsList>
          <TabsTrigger value="chatroom">Chatroom</TabsTrigger>
          <TabsTrigger value="flow">Agent Message Flow</TabsTrigger>
          <TabsTrigger value="diagram">Mermaid Flow Diagram</TabsTrigger>
        </TabsList>

        <TabsContent value="chatroom" className="h-[600px] overflow-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {agents.map(agent => (
              <div key={agent.id} className="border rounded-lg p-4">
                <h3 className="font-semibold mb-3">{agent.id}</h3>
                <div className="space-y-2 h-[500px] overflow-y-auto">
                  {(chatHistories[agent.id] || []).map((msg, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg ${
                        msg.role === 'user'
                          ? 'bg-primary text-primary-foreground ml-4'
                          : 'bg-muted mr-4'
                      }`}
                    >
                      <div className="text-xs font-semibold mb-1">{msg.role}</div>
                      <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="flow" className="h-[600px] overflow-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {agents.map(agent => {
              const agentFlows = flowEntries.filter(entry => entry.agent === agent.id)
              return (
                <div key={agent.id} className="border rounded-lg p-4">
                  <h3 className="font-semibold mb-3">{agent.id}</h3>
                  <div className="space-y-2 h-[500px] overflow-y-auto">
                    {agentFlows.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No messages yet
                      </p>
                    ) : (
                      agentFlows.map((entry, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border ${
                            entry.role === 'system'
                              ? 'bg-accent border-accent-foreground/20'
                              : entry.role === 'user'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          }`}
                        >
                          <div className="text-xs font-semibold mb-1 capitalize">
                            {entry.role}
                          </div>
                          <div className="text-sm whitespace-pre-wrap break-words">
                            {entry.message}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </TabsContent>

        <TabsContent value="diagram" className="h-[600px] overflow-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {agents.map(agent => (
              <div key={agent.id} className="border rounded-lg p-4">
                <h3 className="font-semibold mb-3">{agent.id}</h3>
                <div className="mermaid-container">
                  <Mermaid
                    mermaidCode={getMermaidDiagram(agent.id)}
                    mermaidConfig={{
                      theme: 'default',
                      themeVariables: {
                        fontSize: '14px'
                      }
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Message box */}
      <div className="flex gap-2 mt-auto">
        <Input
          placeholder="Ask once and fan out to selected agents..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          className="flex-1"
        />
        <Button onClick={handleSubmit} disabled={loading}>
          <Send className="w-4 h-4 mr-2" />
          Send
        </Button>
      </div>
    </div>
  )
}
