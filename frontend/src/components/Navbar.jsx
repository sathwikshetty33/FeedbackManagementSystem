// import { Link, useNavigate } from 'react-router-dom'
// import { useAuth } from '../hooks/useAuth.jsx'
// import { useState } from 'react'

// const Navbar = () => {
//   const { user, logout } = useAuth()
//   const navigate = useNavigate()
//   const [isOpen, setIsOpen] = useState(false)

//   const handleLogout = () => {
//     logout()
//     localStorage.removeItem('token')
//     navigate('/login')
//   }

//   const isTeacherOrAdmin = user?.user_type === 'Teacher' || user?.user_type === 'admin'

//   // Determine dashboard path dynamically
//   const getDashboardLink = () => {
//     if (user?.user_type === 'Student') return '/student-dashboard'
//     if (user?.user_type === 'Teacher') return '/teacher-dashboard'
//     return null // no dashboard for others
//   }

//   const dashboardLink = getDashboardLink()

//   return (
//     <nav className="bg-gradient-to-r from-gray-900 to-black border-b border-blue-500/30 shadow-lg shadow-blue-500/10 sticky top-0 z-50 backdrop-blur-xl">
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//         <div className="flex items-center justify-between h-16">
//           {/* Logo */}
//           <Link to="/" className="flex items-center space-x-3 group">
//             <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/50 group-hover:shadow-blue-500/70 transition-all duration-300 transform group-hover:scale-105">
//               <span className="text-white font-bold text-xl">F</span>
//             </div>
//             <span className="text-white font-bold text-xl hidden sm:block tracking-wide group-hover:text-blue-400 transition-colors duration-300">
//               FeedTrack
//             </span>
//           </Link>

//           {/* Desktop Navigation */}
//           <div className="hidden md:flex items-center space-x-1">
//             <Link
//               to="/"
//               className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//             >
//               Home
//             </Link>

//             {user ? (
//               <>
//                 {dashboardLink && (
//                   <Link
//                     to={dashboardLink}
//                     className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                   >
//                     Dashboard
//                   </Link>
//                 )}

//                 {isTeacherOrAdmin && (
//                   <Link
//                     to="/event-dashboard"
//                     className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                   >
//                     Events Dashboard
//                   </Link>
//                 )}

//                 <Link
//                   to="/about"
//                   className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                 >
//                   About
//                 </Link>

//                 <div className="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-700">
//                   <div className="flex items-center space-x-2">
//                     <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-500/50">
//                       {user.name?.[0]?.toUpperCase() || 'U'}
//                     </div>
//                     <span className="text-blue-400 font-medium tracking-wide">
//                       {user.name}
//                     </span>
//                   </div>
//                   <button
//                     onClick={handleLogout}
//                     className="bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg border border-blue-500/30 hover:border-blue-500 transition-all duration-300 font-medium shadow-sm hover:shadow-lg hover:shadow-blue-500/20"
//                   >
//                     Logout
//                   </button>
//                 </div>
//               </>
//             ) : (
//               <>
//                 <Link
//                   to="/login"
//                   className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                 >
//                   Login
//                 </Link>
//                 <Link
//                   to="/signup"
//                   className="ml-2 bg-blue-500 text-white px-6 py-2 rounded-lg font-semibold shadow-lg shadow-blue-500/50 hover:bg-blue-600 hover:shadow-blue-500/70 transition-all duration-300 transform hover:scale-105"
//                 >
//                   Sign Up
//                 </Link>
//               </>
//             )}
//           </div>

//           {/* Mobile Menu Button */}
//           <button
//             onClick={() => setIsOpen(!isOpen)}
//             className="md:hidden text-gray-300 hover:text-blue-400 focus:outline-none transition-colors duration-300 p-2 hover:bg-gray-800/50 rounded-lg"
//             aria-label="Toggle menu"
//           >
//             <svg
//               className="h-6 w-6"
//               fill="none"
//               stroke="currentColor"
//               strokeLinecap="round"
//               strokeLinejoin="round"
//               strokeWidth="2"
//               viewBox="0 0 24 24"
//             >
//               {isOpen ? (
//                 <path d="M6 18L18 6M6 6l12 12" />
//               ) : (
//                 <path d="M4 6h16M4 12h16M4 18h16" />
//               )}
//             </svg>
//           </button>
//         </div>
//       </div>

//       {/* Mobile Menu */}
//       <div
//         className={`md:hidden transition-all duration-300 overflow-hidden ${
//           isOpen ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
//         } bg-black/95 backdrop-blur-xl border-t border-blue-500/20`}
//       >
//         <div className="px-4 pt-3 pb-4 space-y-1">
//           <Link
//             to="/"
//             className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//             onClick={() => setIsOpen(false)}
//           >
//             Home
//           </Link>

//           {user ? (
//             <>
//               {dashboardLink && (
//                 <Link
//                   to={dashboardLink}
//                   className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                   onClick={() => setIsOpen(false)}
//                 >
//                   Dashboard
//                 </Link>
//               )}

//               {isTeacherOrAdmin && (
//                 <Link
//                   to="/event-dashboard"
//                   className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                   onClick={() => setIsOpen(false)}
//                 >
//                   Events Dashboard
//                 </Link>
//               )}

//               <Link
//                 to="/about"
//                 className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                 onClick={() => setIsOpen(false)}
//               >
//                 About
//               </Link>

//               <div className="px-4 py-3 border-t border-gray-800 mt-2 pt-3">
//                 <div className="flex items-center space-x-2 mb-3">
//                   <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-500/50">
//                     {user.name?.[0]?.toUpperCase() || 'U'}
//                   </div>
//                   <span className="text-blue-400 font-medium tracking-wide">
//                     {user.name}
//                   </span>
//                 </div>
//                 <button
//                   onClick={() => {
//                     handleLogout()
//                     setIsOpen(false)
//                   }}
//                   className="w-full px-4 py-3 text-left bg-gray-900 hover:bg-gray-800 text-white rounded-lg border border-blue-500/30 hover:border-blue-500 transition-all duration-300 font-medium"
//                 >
//                   Logout
//                 </button>
//               </div>
//             </>
//           ) : (
//             <>
//               <Link
//                 to="/login"
//                 className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium"
//                 onClick={() => setIsOpen(false)}
//               >
//                 Login
//               </Link>
//               <Link
//                 to="/signup"
//                 className="block px-4 py-3 text-center bg-blue-500 text-white rounded-lg font-semibold shadow-lg shadow-blue-500/50 hover:bg-blue-600 hover:shadow-blue-500/70 transition-all duration-300"
//                 onClick={() => setIsOpen(false)}
//               >
//                 Sign Up
//               </Link>
//             </>
//           )}
//         </div>
//       </div>
//     </nav>
//   )
// }

// export default Navbar
import { NavLink, useNavigate } from 'react-router-dom'
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

  const isTeacherOrAdmin = user?.user_type === 'Teacher' || user?.user_type === 'admin'

  // Determine dashboard path dynamically
  const getDashboardLink = () => {
    if (user?.user_type === 'Student') return '/student-dashboard'
    if (user?.user_type === 'Teacher') return '/teacher-dashboard'
    return null // no dashboard for others
  }

  const dashboardLink = getDashboardLink()

  // Common NavLink class generator
  const navClass = ({ isActive }) =>
    isActive
      ? 'px-4 py-2 text-blue-400 font-semibold bg-gray-800/70 rounded-lg transition-all duration-300'
      : 'px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-800/50 rounded-lg transition-all duration-300 font-medium'

  return (
    <nav className="bg-gradient-to-r from-gray-900 to-black border-b border-blue-500/30 shadow-lg shadow-blue-500/10 sticky top-0 z-50 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <NavLink to="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/50 group-hover:shadow-blue-500/70 transition-all duration-300 transform group-hover:scale-105">
              <span className="text-white font-bold text-xl">F</span>
            </div>
            <span className="text-white font-bold text-xl hidden sm:block tracking-wide group-hover:text-blue-400 transition-colors duration-300">
              FeedTrack
            </span>
          </NavLink>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            <NavLink to="/" className={navClass}>
              Home
            </NavLink>

            {user ? (
              <>
                {dashboardLink && (
                  <NavLink to={dashboardLink} className={navClass}>
                    Dashboard
                  </NavLink>
                )}

                {isTeacherOrAdmin && (
                  <NavLink to="/event-dashboard" className={navClass}>
                    Events Dashboard
                  </NavLink>
                )}

                <NavLink to="/about" className={navClass}>
                  About
                </NavLink>

                <div className="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-700">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-500/50">
                      {user.name?.[0]?.toUpperCase() || 'U'}
                    </div>
                    <span className="text-blue-400 font-medium tracking-wide">
                      {user.name}
                    </span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg border border-blue-500/30 hover:border-blue-500 transition-all duration-300 font-medium shadow-sm hover:shadow-lg hover:shadow-blue-500/20"
                  >
                    Logout
                  </button>
                </div>
              </>
            ) : (
              <>
                <NavLink to="/login" className={navClass}>
                  Login
                </NavLink>
                <NavLink
                  to="/signup"
                  className={({ isActive }) =>
                    isActive
                      ? 'ml-2 bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold shadow-lg shadow-blue-500/70 transition-all duration-300 transform scale-105'
                      : 'ml-2 bg-blue-500 text-white px-6 py-2 rounded-lg font-semibold shadow-lg shadow-blue-500/50 hover:bg-blue-600 hover:shadow-blue-500/70 transition-all duration-300 transform hover:scale-105'
                  }
                >
                  Sign Up
                </NavLink>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-gray-300 hover:text-blue-400 focus:outline-none transition-colors duration-300 p-2 hover:bg-gray-800/50 rounded-lg"
            aria-label="Toggle menu"
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
        } bg-black/95 backdrop-blur-xl border-t border-blue-500/20`}
      >
        <div className="px-4 pt-3 pb-4 space-y-1">
          <NavLink
            to="/"
            className={navClass}
            onClick={() => setIsOpen(false)}
          >
            Home
          </NavLink>

          {user ? (
            <>
              {dashboardLink && (
                <NavLink
                  to={dashboardLink}
                  className={navClass}
                  onClick={() => setIsOpen(false)}
                >
                  Dashboard
                </NavLink>
              )}

              {isTeacherOrAdmin && (
                <NavLink
                  to="/event-dashboard"
                  className={navClass}
                  onClick={() => setIsOpen(false)}
                >
                  Events Dashboard
                </NavLink>
              )}

              <NavLink
                to="/about"
                className={navClass}
                onClick={() => setIsOpen(false)}
              >
                About
              </NavLink>

              <div className="px-4 py-3 border-t border-gray-800 mt-2 pt-3">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-500/50">
                    {user.name?.[0]?.toUpperCase() || 'U'}
                  </div>
                  <span className="text-blue-400 font-medium tracking-wide">
                    {user.name}
                  </span>
                </div>
                <button
                  onClick={() => {
                    handleLogout()
                    setIsOpen(false)
                  }}
                  className="w-full px-4 py-3 text-left bg-gray-900 hover:bg-gray-800 text-white rounded-lg border border-blue-500/30 hover:border-blue-500 transition-all duration-300 font-medium"
                >
                  Logout
                </button>
              </div>
            </>
          ) : (
            <>
              <NavLink
                to="/login"
                className={navClass}
                onClick={() => setIsOpen(false)}
              >
                Login
              </NavLink>
              <NavLink
                to="/signup"
                className={({ isActive }) =>
                  isActive
                    ? 'block px-4 py-3 text-center bg-blue-600 text-white rounded-lg font-semibold shadow-lg shadow-blue-500/70 transition-all duration-300'
                    : 'block px-4 py-3 text-center bg-blue-500 text-white rounded-lg font-semibold shadow-lg shadow-blue-500/50 hover:bg-blue-600 hover:shadow-blue-500/70 transition-all duration-300'
                }
                onClick={() => setIsOpen(false)}
              >
                Sign Up
              </NavLink>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar
