import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth.jsx'
import { useState } from 'react'

const Navbar = () => {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [isOpen, setIsOpen] = useState(false)

  const handleLogout = () => {
    logout()
    localStorage.removeItem('token')
    navigate('/login')
  }

  return (
    <nav className="bg-black/80 backdrop-blur-lg border-b border-blue-500/30 shadow-lg shadow-blue-500/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/50 group-hover:shadow-blue-500/70 transition-all duration-300">
              <span className="text-white font-bold text-xl">F</span>
            </div>
            <span className="text-white font-semibold text-xl hidden sm:block tracking-wide group-hover:text-blue-400 transition-colors duration-300">
              FeedTrack
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              to="/"
              className="text-gray-300 hover:text-blue-400 transition-all duration-300 hover:scale-105"
            >
              Home
            </Link>

            {user ? (
              <>
                <Link
                  to="/dashboard"
                  className="text-gray-300 hover:text-blue-400 transition-all duration-300 hover:scale-105"
                >
                  Dashboard
                </Link>
                <Link
                  to="/about"
                  className="text-gray-300 hover:text-blue-400 transition-all duration-300 hover:scale-105"
                >
                  About
                </Link>

                <div className="flex items-center space-x-4">
                  <span className="text-blue-400 font-medium tracking-wide">
                    {user.name}
                  </span>
                  <button
                    onClick={handleLogout}
                    className="bg-blue-500/20 text-blue-400 px-4 py-2 rounded-lg border border-blue-500/50 hover:bg-blue-600/30 hover:shadow-lg hover:shadow-blue-500/50 transition-all duration-300 hover:scale-105"
                  >
                    Logout
                  </button>
                </div>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-gray-300 hover:text-blue-400 transition-all duration-300 hover:scale-105"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="bg-blue-500 text-white px-6 py-2 rounded-lg font-medium shadow-md shadow-blue-500/50 hover:bg-blue-600 hover:shadow-blue-500/70 transition-all duration-300 hover:scale-105"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-gray-300 hover:text-blue-400 focus:outline-none transition-colors duration-300"
          >
            <svg
              className="h-6 w-6"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              {isOpen ? (
                <path d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`md:hidden transition-all duration-300 overflow-hidden ${
          isOpen ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
        } bg-gray-900/95 backdrop-blur-xl border-t border-blue-500/20`}
      >
        <div className="px-4 pt-3 pb-4 space-y-2">
          <Link
            to="/"
            className="block px-3 py-2 text-gray-300 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-all duration-300"
            onClick={() => setIsOpen(false)}
          >
            Home
          </Link>

          {user ? (
            <>
              <Link
                to="/dashboard"
                className="block px-3 py-2 text-gray-300 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-all duration-300"
                onClick={() => setIsOpen(false)}
              >
                Dashboard
              </Link>
              <Link
                to="/about"
                className="block px-3 py-2 text-gray-300 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-all duration-300"
                onClick={() => setIsOpen(false)}
              >
                About
              </Link>
              <div className="px-3 py-1 text-blue-400 font-medium tracking-wide">
                {user.name}
              </div>
              <button
                onClick={() => {
                  handleLogout()
                  setIsOpen(false)
                }}
                className="w-full text-left px-3 py-2 text-red-400 hover:text-white hover:bg-red-500/20 rounded-lg transition-all duration-300"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className="block px-3 py-2 text-gray-300 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-all duration-300"
                onClick={() => setIsOpen(false)}
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="block px-3 py-2 text-center bg-blue-500 text-white rounded-lg hover:bg-blue-600 shadow-md shadow-blue-500/50 hover:shadow-blue-500/70 transition-all duration-300"
                onClick={() => setIsOpen(false)}
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar
