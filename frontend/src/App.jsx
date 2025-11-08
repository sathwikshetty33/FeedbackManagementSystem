import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'
import Intro from './pages/Intro'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/StudentDashboard'
import StudentDashboard from './pages/StudentDashboard'
import TeacherDashboard from './pages/TeacherDashboard'
import AdminDashboard from './pages/AdminDashboard'
import About from './pages/About'
import EventsDashboard from './pages/EventDashboard'
import Footer from './pages/Footer'
import EventChat from './pages/EventChat'
function App() {
  return (
    <>
      <Navbar />
      <div className="min-h-screen bg-black">
        <Routes>
          <Route path="/" element={<Intro />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          
          {/* Student Dashboard */}
          <Route
            path="/student-dashboard"
            element={
              <ProtectedRoute allowedUserTypes={['Student']}>
                <StudentDashboard />
              </ProtectedRoute>
            }
          />
          
          {/* Teacher Dashboard */}
          <Route
            path="/teacher-dashboard"
            element={
              <ProtectedRoute allowedUserTypes={['Teacher']}>
                <TeacherDashboard />
              </ProtectedRoute>
            }
          />
          
          {/* Admin Dashboard
          <Route
            path="/admin-dashboard"
            element={
              <ProtectedRoute allowedUserTypes={['Superuser']}>
                <AdminDashboard />
              </ProtectedRoute>
            }
          /> */}
          {/* Admin Dashboard */}
          <Route
            path="/event-dashboard"
            element={
              <ProtectedRoute allowedUserTypes={['Superuser']}>
                <EventsDashboard />
              </ProtectedRoute>
            }
          />
          {/* General Dashboard (fallback)
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          /> */}
          
          <Route
            path="/about"
            element={
              <ProtectedRoute>
                <About />
              </ProtectedRoute>
            }
          />
          <Route path="/chat" element={<EventChat />} />
        </Routes>
        <Footer />
      </div>
      
    </>
  )
}

export default App