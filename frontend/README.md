# AI Agent Lab Frontend

Modern React frontend for the AI Agent Lab platform, built with Vite, TypeScript, and shadcn/ui components.

## Features

- **Multi-Agent Chat Interface**: Chat with multiple AI agents simultaneously
- **Agent Selection**: Toggle which agents respond to your messages
- **System Prompts**: Share system instructions across all active agents
- **Message Flow Tracking**: View the complete conversation flow across agents
- **Mermaid Diagrams**: Visualize each agent's processing workflow
- **Responsive Design**: Built with Tailwind CSS and shadcn/ui components

## Technology Stack

- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **Mermaid** - Diagram rendering
- **Radix UI** - Accessible primitives

## Getting Started

### Prerequisites

- Node.js 18+ or pnpm
- Backend server running on port 11111 (default)

### Installation

```bash
# Install dependencies
pnpm install

# Copy environment file
cp .env.example .env

# Edit .env if needed (default: http://localhost:11111)
```

### Development

```bash
# Start development server
pnpm dev
```

Visit `http://localhost:5173` to see the application.

### Build

```bash
# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_BACKEND_URL=http://localhost:11111
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── ui/          # shadcn/ui components
│   ├── lib/
│   │   └── utils.ts     # Utility functions
│   ├── App.tsx          # Main application component
│   ├── App.css          # Application styles
│   ├── index.css        # Tailwind & global styles
│   └── main.tsx         # Entry point
├── public/              # Static assets
├── tailwind.config.js   # Tailwind configuration
├── vite.config.ts       # Vite configuration
└── package.json
```

## Available Agents

- **rag-faiss**: RAG agent with Faiss vector store
- **graphrag-neo4j**: GraphRAG agent with Neo4j graph database
- **cognee**: Cognee agent for knowledge processing
- **leann**: Lightweight vector database with graph-based recomputation

## Usage

1. **Select Agents**: Check the agents you want to interact with
2. **Add System Prompt** (optional): Provide instructions for all agents
3. **Send Messages**: Type your message and click Send or press Enter
4. **View Responses**: See responses from each agent in the Chatroom tab
5. **Track Flow**: Monitor message flow in the Agent Message Flow tab
6. **View Diagrams**: Explore agent workflows in the Mermaid Flow Diagram tab

## Keyboard Shortcuts

- `Enter`: Send message
- `Shift + Enter`: New line in message input

## Development Notes

- The frontend automatically falls back to default agents if the backend is unavailable
- All agent communication goes through the `/chat` endpoint
- Session state is maintained per agent
- Mermaid diagrams are rendered on component mount

## License

MIT
