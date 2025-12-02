import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Plus, Trash2, Edit } from 'lucide-react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:11111'

interface SystemPrompt {
  id: string
  name: string
  content: string
  user_id?: string | null
  created_at: string
  updated_at: string
}

export function SystemPrompts() {
  const [prompts, setPrompts] = useState<SystemPrompt[]>([])
  const [isCreating, setIsCreating] = useState(false)
  const [newPrompt, setNewPrompt] = useState({ name: '', content: '' })
  const [editingId, setEditingId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Load prompts from backend
  useEffect(() => {
    loadPrompts()
  }, [])

  const loadPrompts = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/prompts`)
      const data = await response.json()
      setPrompts(data)
    } catch (err) {
      console.error('Failed to load prompts:', err)
    }
  }

  const handleCreate = async () => {
    if (newPrompt.name.trim() && newPrompt.content.trim()) {
      setLoading(true)
      try {
        const response = await fetch(`${BACKEND_URL}/prompts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: newPrompt.name,
            content: newPrompt.content,
            user_id: null
          })
        })

        if (response.ok) {
          await loadPrompts()
          setNewPrompt({ name: '', content: '' })
          setIsCreating(false)
        } else {
          console.error('Failed to create prompt')
        }
      } catch (err) {
        console.error('Error creating prompt:', err)
      } finally {
        setLoading(false)
      }
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this prompt?')) return

    setLoading(true)
    try {
      const response = await fetch(`${BACKEND_URL}/prompts/${id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        await loadPrompts()
      } else {
        console.error('Failed to delete prompt')
      }
    } catch (err) {
      console.error('Error deleting prompt:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (id: string) => {
    setEditingId(id)
    const prompt = prompts.find(p => p.id === id)
    if (prompt) {
      setNewPrompt({ name: prompt.name, content: prompt.content })
    }
  }

  const handleUpdate = async () => {
    if (editingId && newPrompt.name.trim() && newPrompt.content.trim()) {
      setLoading(true)
      try {
        const response = await fetch(`${BACKEND_URL}/prompts/${editingId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: newPrompt.name,
            content: newPrompt.content
          })
        })

        if (response.ok) {
          await loadPrompts()
          setNewPrompt({ name: '', content: '' })
          setEditingId(null)
        } else {
          console.error('Failed to update prompt')
        }
      } catch (err) {
        console.error('Error updating prompt:', err)
      } finally {
        setLoading(false)
      }
    }
  }

  const handleCancel = () => {
    setNewPrompt({ name: '', content: '' })
    setIsCreating(false)
    setEditingId(null)
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold mb-2">System Prompts</h1>
        <p className="text-muted-foreground">
          Create and manage reusable system prompts for your AI agents
        </p>
      </div>

      <div className="mb-6">
        <Button onClick={() => setIsCreating(true)} disabled={isCreating || editingId !== null || loading}>
          <Plus className="w-4 h-4 mr-2" />
          New Prompt
        </Button>
      </div>

      {/* Create/Edit Form */}
      {(isCreating || editingId) && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>{editingId ? 'Edit Prompt' : 'Create New Prompt'}</CardTitle>
            <CardDescription>
              {editingId ? 'Update your system prompt' : 'Add a new system prompt to your library'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="prompt-name">Name</Label>
              <Input
                id="prompt-name"
                placeholder="e.g., Professional Assistant"
                value={newPrompt.name}
                onChange={(e) => setNewPrompt({ ...newPrompt, name: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="prompt-content">Content</Label>
              <Textarea
                id="prompt-content"
                placeholder="Enter your system prompt..."
                rows={6}
                value={newPrompt.content}
                onChange={(e) => setNewPrompt({ ...newPrompt, content: e.target.value })}
              />
            </div>
          </CardContent>
          <CardFooter className="gap-2">
            <Button onClick={editingId ? handleUpdate : handleCreate} disabled={loading}>
              {loading ? 'Saving...' : editingId ? 'Update' : 'Create'}
            </Button>
            <Button variant="outline" onClick={handleCancel} disabled={loading}>
              Cancel
            </Button>
          </CardFooter>
        </Card>
      )}

      {/* Prompts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {prompts.map((prompt) => (
          <Card key={prompt.id} className="flex flex-col">
            <CardHeader>
              <CardTitle className="text-lg">{prompt.name}</CardTitle>
              <CardDescription>
                Created {new Date(prompt.created_at).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1">
              <p className="text-sm text-muted-foreground line-clamp-4">
                {prompt.content}
              </p>
            </CardContent>
            <CardFooter className="gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleEdit(prompt.id)}
                disabled={editingId !== null || isCreating || loading}
              >
                <Edit className="w-4 h-4 mr-1" />
                Edit
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => handleDelete(prompt.id)}
                disabled={editingId !== null || isCreating || loading}
              >
                <Trash2 className="w-4 h-4 mr-1" />
                Delete
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      {prompts.length === 0 && !isCreating && (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No system prompts yet</p>
          <Button onClick={() => setIsCreating(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Create Your First Prompt
          </Button>
        </div>
      )}
    </div>
  )
}
