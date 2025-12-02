import { Link, useLocation } from 'react-router-dom'
import { Home, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ThemeToggle } from './theme-toggle'

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'System Prompts', href: '/prompts', icon: FileText },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <div className="flex h-screen w-64 flex-col border-r bg-card">
      <div className="flex h-16 items-center border-b px-6">
        <h1 className="text-xl font-bold">AI Agent Lab</h1>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>
      <div className="border-t p-4 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Theme
          </p>
          <ThemeToggle />
        </div>
        <p className="text-xs text-muted-foreground">
          Multi-agent platform for AI experimentation
        </p>
      </div>
    </div>
  )
}
