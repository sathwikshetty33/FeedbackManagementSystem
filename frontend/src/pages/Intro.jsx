import { Link } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth.jsx'

const Intro = () => {
  const { user } = useAuth()

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <div className="max-w-4xl mx-auto text-center">
        {/* Animated gradient orb */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse"></div>
        
        <div className="relative z-10">
          <h1 className="text-6xl md:text-8xl font-bold text-white mb-6 animate-fade-in">
            Welcome to
            <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-purple-400 text-transparent bg-clip-text animate-gradient">
              {' '}AppName
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-2xl mx-auto">
            Experience the future of web applications with our cutting-edge platform. 
            Built with modern technologies for exceptional performance.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            {user ? (
              <>
                <Link
                  to="/dashboard"
                  className="w-full sm:w-auto bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-purple-500/50 hover:scale-105"
                >
                  Go to Dashboard
                </Link>
                <Link
                  to="/about"
                  className="w-full sm:w-auto bg-slate-800/50 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-slate-700/50 transition-all duration-200 border border-purple-500/30"
                >
                  Learn More
                </Link>
              </>
            ) : (
              <>
                <Link
                  to="/signup"
                  className="w-full sm:w-auto bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-purple-500/50 hover:scale-105"
                >
                  Get Started
                </Link>
                <Link
                  to="/login"
                  className="w-full sm:w-auto bg-slate-800/50 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-slate-700/50 transition-all duration-200 border border-purple-500/30"
                >
                  Sign In
                </Link>
              </>
            )}
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-8 mt-24">
            <div className="bg-slate-800/30 backdrop-blur-lg p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-200">
              <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Lightning Fast</h3>
              <p className="text-gray-400">Built with Vite for blazing fast performance</p>
            </div>

            <div className="bg-slate-800/30 backdrop-blur-lg p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-200">
              <div className="w-12 h-12 bg-pink-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Secure</h3>
              <p className="text-gray-400">Protected routes with authentication</p>
            </div>

            <div className="bg-slate-800/30 backdrop-blur-lg p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-200">
              <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Modern Design</h3>
              <p className="text-gray-400">Beautiful UI with Tailwind CSS</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Intro