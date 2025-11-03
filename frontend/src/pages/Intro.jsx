import { Link } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth.jsx'
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const Intro = () => {
  const { user } = useAuth()
  const navigate = useNavigate()

useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    const userType = user?.user_type
    if (token && userType) {
      // Redirect based on user type
      switch(userType) {
        case 'Student':
          navigate('/student-dashboard')
          break
        case 'Teacher':
          navigate('/teacher-dashboard')
          break
        case 'Superuser':
          navigate('/admin-dashboard')
          break
        default:
          navigate('/dashboard')
      }
    }
  }, [navigate, user])
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 bg-black">
      <div className="max-w-4xl mx-auto text-center">
        {/* Animated neon blue orb */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-500/30 rounded-full blur-3xl animate-pulse"></div>
        
        <div className="relative z-10">
          <h1 className="text-6xl md:text-8xl font-bold text-white mb-6 animate-fade-in">
            Transform
            <span className="bg-gradient-to-r from-blue-400 via-blue-500 to-blue-400 text-transparent bg-clip-text">
              {' '}Feedback
            </span>
            <br />
            <span className="text-5xl md:text-6xl">Into Intelligence</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-2xl mx-auto">
            AI-powered feedback analysis platform that transforms raw data into actionable insights. 
            Built for educators and institutions to make data-driven decisions.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            {user ? (
              <>
                <Link
                  to="/dashboard"
                  className="w-full sm:w-auto bg-blue-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-600 transition-all duration-200 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 hover:scale-105"
                >
                  Go to Dashboard
                </Link>
                <Link
                  to="/about"
                  className="w-full sm:w-auto bg-black text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-gray-900 transition-all duration-200 border-2 border-blue-500"
                >
                  Learn More
                </Link>
              </>
            ) : (
              <>
                <Link
                  to="/signup"
                  className="w-full sm:w-auto bg-blue-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-600 transition-all duration-200 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 hover:scale-105"
                >
                  Get Started
                </Link>
                <Link
                  to="/login"
                  className="w-full sm:w-auto bg-black text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-gray-900 transition-all duration-200 border-2 border-blue-500"
                >
                  Sign In
                </Link>
              </>
            )}
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-8 mt-24">
            <div className="bg-gray-900 backdrop-blur-lg p-6 rounded-xl border-2 border-blue-500/50 hover:border-blue-500 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/30">
              <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto border border-blue-500/50">
                <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">AI-Powered Analysis</h3>
              <p className="text-gray-400">Advanced sentiment analysis and pattern recognition using LangChain and Ollama</p>
            </div>

            <div className="bg-gray-900 backdrop-blur-lg p-6 rounded-xl border-2 border-blue-500/50 hover:border-blue-500 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/30">
              <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto border border-blue-500/50">
                <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Real-Time Processing</h3>
              <p className="text-gray-400">FastAPI-powered async analysis with concurrent task processing</p>
            </div>

            <div className="bg-gray-900 backdrop-blur-lg p-6 rounded-xl border-2 border-blue-500/50 hover:border-blue-500 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/30">
              <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto border border-blue-500/50">
                <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Multi-User Support</h3>
              <p className="text-gray-400">Role-based access for admins, teachers, students, and public users</p>
            </div>
          </div>

          {/* Additional Info Section */}
          <div className="mt-20 bg-gray-900 backdrop-blur-lg p-8 rounded-2xl border-2 border-blue-500/50">
            <h2 className="text-3xl font-bold text-white mb-4">
              Why Choose Our Platform?
            </h2>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500/20 rounded flex items-center justify-center flex-shrink-0 mt-1 border border-blue-500/50">
                  <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">5NF Database Design</h4>
                  <p className="text-gray-400 text-sm">Fully normalized schema ensuring data integrity and optimal performance</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500/20 rounded flex items-center justify-center flex-shrink-0 mt-1 border border-blue-500/50">
                  <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Vector Search with FAISS</h4>
                  <p className="text-gray-400 text-sm">Efficient semantic search for contextual feedback analysis</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500/20 rounded flex items-center justify-center flex-shrink-0 mt-1 border border-blue-500/50">
                  <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Dockerized Architecture</h4>
                  <p className="text-gray-400 text-sm">Scalable multi-container setup with nginx reverse proxy</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500/20 rounded flex items-center justify-center flex-shrink-0 mt-1 border border-blue-500/50">
                  <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">RAG-Based Insights</h4>
                  <p className="text-gray-400 text-sm">Contextual understanding with retrieval-augmented generation</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Intro