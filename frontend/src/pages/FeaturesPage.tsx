import { Link } from 'react-router-dom';

export default function FeaturesPage() {
  const features = [
    {
      title: 'Text Simplification',
      description: 'Simplify educational content for different grade levels using AI',
      icon: '📝',
      path: '/simplify',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      title: 'Multi-Language Translation',
      description: 'Translate content into 10+ Indian languages',
      icon: '🌐',
      path: '/translate',
      color: 'from-purple-500 to-pink-500'
    },
    {
      title: 'Text-to-Speech',
      description: 'Generate natural audio from text in multiple languages',
      icon: '🔊',
      path: '/tts',
      color: 'from-green-500 to-emerald-500'
    },
    {
      title: 'Content Validation',
      description: 'Validate content against NCERT standards',
      icon: '✓',
      path: '/validate',
      color: 'from-orange-500 to-red-500'
    },
    {
      title: 'Q&A System',
      description: 'Ask questions about uploaded documents using AI',
      icon: '💬',
      path: '/qa',
      color: 'from-indigo-500 to-purple-500'
    },
    {
      title: 'Full Pipeline',
      description: 'Process content through all steps automatically',
      icon: '⚡',
      path: '/upload',
      color: 'from-yellow-500 to-orange-500'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
            ShikshaSetu Features
          </h1>
          <p className="text-xl text-gray-600">
            Comprehensive AI-powered educational content processing tools
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => (
            <Link
              key={feature.path}
              to={feature.path}
              className="group"
            >
              <div className="h-full glass-card p-8 hover:shadow-2xl transition-all duration-300 transform group-hover:-translate-y-2">
                <div className={`text-6xl mb-4 p-4 rounded-2xl bg-gradient-to-br ${feature.color} inline-block`} aria-hidden="true">
                  <span className="filter drop-shadow-lg">{feature.icon}</span>
                </div>
                <h2 className="text-2xl font-bold mb-3 text-gray-900 group-hover:text-primary-600 transition-colors">
                  {feature.title}
                </h2>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
                <div className="mt-6 flex items-center text-primary-600 font-semibold group-hover:translate-x-2 transition-transform">
                  Try it now
                  <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-16 glass-card p-8 text-center">
          <h3 className="text-2xl font-bold mb-4 text-gray-900">Need Help Getting Started?</h3>
          <p className="text-gray-600 mb-6">
            Check out our documentation or contact support for assistance
          </p>
          <div className="flex gap-4 justify-center">
            <Link to="/dashboard" className="btn-primary">
              Go to Dashboard
            </Link>
            <Link to="/library" className="btn-secondary">
              View Library
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
