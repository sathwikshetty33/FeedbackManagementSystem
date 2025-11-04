import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuth from '../hooks/useAuth';

const EventsDashboard = () => {
  const { getToken, API_URL } = useAuth();
  const navigate = useNavigate();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [insightsModal, setInsightsModal] = useState({ show: false, eventId: null, eventName: '', loading: false });

  useEffect(() => {
    const token = getToken();
    if (!token) {
      navigate('/login');
      return;
    }
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await fetch(`${API_URL}/events/all-events/`, {
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setEvents(data);
      } else {
        throw new Error('Failed to fetch events');
      }
    } catch (error) {
      console.error('Error:', error);
      showNotification('Error loading events', 'error');
    } finally {
      setLoading(false);
    }
  };

  const deleteEvent = async (eventId) => {
    if (!window.confirm('Are you sure you want to delete this event?')) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/events/events/${eventId}/`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        showNotification('Event deleted successfully', 'success');
        loadEvents();
      } else {
        throw new Error('Failed to delete event');
      }
    } catch (error) {
      console.error('Error:', error);
      showNotification('Error deleting event', 'error');
    }
  };

  const generateInsights = async (eventId, eventName) => {
    setInsightsModal({ show: true, eventId, eventName, loading: true });

    try {
      const response = await fetch(`${API_URL}/insights/generate-insights/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Token ${getToken()}`
        },
        body: JSON.stringify({ event_id: eventId })
      });

      if (response.status === 200) {
        setInsightsModal(prev => ({ ...prev, loading: false, success: true }));
      } else {
        setInsightsModal(prev => ({ ...prev, loading: false, success: false }));
      }
    } catch (error) {
      console.error('Error:', error);
      setInsightsModal(prev => ({ ...prev, loading: false, success: false }));
    }
  };

  const openChat = (eventId, eventName) => {
    const currentUrl = window.location.href;
    const chatUrl = `/chat?dashboard_url=${encodeURIComponent(currentUrl)}&event_id=${eventId}&event_name=${encodeURIComponent(eventName)}`;
    navigate(chatUrl);
  };

  const showNotification = (message, type) => {
    setNotification({ show: true, message, type });
    setTimeout(() => {
      setNotification({ show: false, message: '', type: '' });
    }, 3000);
  };

  const getVisibilityLabel = (visibility) => {
    const map = {
      '1': 'Semester 1', '2': 'Semester 2', '3': 'Semester 3',
      '4': 'Semester 4', '5': 'Semester 5', '6': 'Semester 6',
      '7': 'Semester 7', '8': 'Semester 8',
      'anyone': 'Anyone', 'teachers': 'Teachers Only'
    };
    return map[visibility] || visibility;
  };

  const closeModal = () => {
    setInsightsModal({ show: false, eventId: null, eventName: '', loading: false });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Notification */}
      {notification.show && (
        <div className={`fixed top-5 right-5 px-6 py-3 rounded-lg shadow-2xl z-50 ${
          notification.type === 'success' 
            ? 'bg-blue-500 shadow-blue-500/50' 
            : 'bg-red-500 shadow-red-500/50'
        } animate-fade-in`}>
          {notification.message}
        </div>
      )}

      {/* Insights Modal */}
      {insightsModal.show && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg border border-blue-500/30 shadow-2xl shadow-blue-500/20 max-w-md w-full">
            <div className="flex justify-between items-center p-6 border-b border-gray-800">
              <h3 className="text-xl font-bold text-white">Generate Insights: {insightsModal.eventName}</h3>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-white text-2xl transition-colors"
              >
                ×
              </button>
            </div>
            
            <div className="p-6 text-center">
              {insightsModal.loading ? (
                <>
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                  <p className="text-gray-300">Generating insights...</p>
                </>
              ) : insightsModal.success ? (
                <>
                  <div className="text-6xl mb-4">✅</div>
                  <h3 className="text-2xl font-bold text-green-400 mb-2">Success!</h3>
                  <p className="text-gray-300">Insights will be sent through email</p>
                </>
              ) : (
                <>
                  <div className="text-6xl mb-4">❌</div>
                  <h3 className="text-2xl font-bold text-red-400 mb-2">Error</h3>
                  <p className="text-gray-300">Some error has occurred</p>
                </>
              )}
            </div>

            <div className="p-6 border-t border-gray-800 flex justify-center">
              <button
                onClick={closeModal}
                className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
          <h2 className="text-3xl font-bold text-white">Events Management</h2>
          <button
            onClick={() => navigate('/admin-create')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70"
          >
            Create New Event
          </button>
        </div>

        {/* Events Table */}
        {events.length === 0 ? (
          <div className="bg-gradient-to-br from-gray-900 to-black rounded-lg border border-blue-500/30 p-12 text-center shadow-xl shadow-blue-500/10">
            <div className="text-6xl mb-4">📅</div>
            <h3 className="text-2xl font-bold text-white mb-2">No events found</h3>
            <p className="text-gray-400 mb-6">Create your first event to get started.</p>
            <button
              onClick={() => navigate('/admin-create')}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50"
            >
              Create New Event
            </button>
          </div>
        ) : (
          <div className="bg-gray-900 rounded-lg border border-blue-500/30 shadow-xl shadow-blue-500/10 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gradient-to-r from-blue-600 to-blue-500">
                    <th className="px-6 py-4 text-left text-white font-semibold">ID</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">Name</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">Start Time</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">End Time</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">Visibility</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event, index) => (
                    <tr 
                      key={event.id}
                      className={`border-b border-gray-800 hover:bg-gray-800/50 transition-colors ${
                        index % 2 === 0 ? 'bg-black/20' : 'bg-black/40'
                      }`}
                    >
                      <td className="px-6 py-4 text-gray-300">{event.id}</td>
                      <td className="px-6 py-4 text-white font-medium">{event.name}</td>
                      <td className="px-6 py-4 text-gray-300">
                        {new Date(event.start_time).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-gray-300">
                        {new Date(event.end_time).toLocaleString()}
                      </td>
                      <td className="px-6 py-4">
                        <span className="inline-block px-3 py-1 bg-blue-500/20 text-blue-400 rounded text-sm border border-blue-500/30">
                          {getVisibilityLabel(event.visibility)}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={() => generateInsights(event.id, event.name)}
                            className="bg-blue-500/20 text-blue-400 border border-blue-500/50 px-3 py-1.5 rounded text-sm hover:bg-blue-500/30 transition-all duration-300"
                          >
                            Insights
                          </button>
                          <button
                            onClick={() => openChat(event.id, event.name)}
                            className="bg-gray-700 text-gray-300 border border-gray-600 px-3 py-1.5 rounded text-sm hover:bg-gray-600 transition-all duration-300"
                          >
                            Chat
                          </button>
                          <button
                            onClick={() => navigate(`/admin-create/${event.id}`)}
                            className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/50 px-3 py-1.5 rounded text-sm hover:bg-yellow-500/30 transition-all duration-300"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => deleteEvent(event.id)}
                            className="bg-red-500/20 text-red-400 border border-red-500/50 px-3 py-1.5 rounded text-sm hover:bg-red-500/30 transition-all duration-300"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-gradient-to-r from-gray-900 to-black border-t border-blue-500/30 mt-16 py-12 px-4">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-bold text-blue-400 mb-4">About FeedTrack</h3>
            <p className="text-gray-400 leading-relaxed">
              FeedTrack is a comprehensive feedback management platform tailored for department events. 
              It empowers organizers to create tailored feedback forms, collect valuable responses, 
              and gain actionable insights—enhancing the quality and impact of every event.
            </p>
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-blue-400 mb-4">Useful Links</h3>
            <ul className="space-y-2">
              <li><a href="/" className="text-gray-400 hover:text-blue-400 transition-colors">Home</a></li>
              <li><a href="/admin-create" className="text-gray-400 hover:text-blue-400 transition-colors">Create Feedback Form</a></li>
              <li><a href="/admin-dashboard" className="text-gray-400 hover:text-blue-400 transition-colors">Analyze Feedback</a></li>
              <li><a href="#" className="text-gray-400 hover:text-blue-400 transition-colors">Contact Us</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-blue-400 mb-4">Contact</h3>
            <div className="space-y-2 text-gray-400">
              <p className="flex items-center gap-2">
                <span>📧</span> CodeZero@gmail.com
              </p>
              <p className="flex items-center gap-2">
                <span>📱</span> +91 6678898997
              </p>
              <div className="flex gap-3 mt-4">
                <a 
                  href="https://www.instagram.com/codezeroaiml/" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white hover:bg-blue-600 transition-all duration-300 shadow-lg shadow-blue-500/50"
                >
                  i
                </a>
                <a 
                  href="https://www.linkedin.com/in/code-zero-4a3406334" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white hover:bg-blue-600 transition-all duration-300 shadow-lg shadow-blue-500/50"
                >
                  in
                </a>
              </div>
            </div>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto mt-8 pt-8 border-t border-gray-800 text-center text-gray-400 text-sm">
          © 2024 FeedTrack. All rights reserved. | Designed by CodeZero Team
        </div>
      </footer>

      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default EventsDashboard;