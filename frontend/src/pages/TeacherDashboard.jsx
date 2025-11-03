import { useState, useEffect } from 'react';
import useAuth from '../hooks/useAuth';

const TeacherDashboard = () => {
  const { user, logout, getToken, API_URL } = useAuth();
  const [activeTab, setActiveTab] = useState('available');
  const [filter, setFilter] = useState('all');
  const [events, setEvents] = useState([]);
  const [attendedEvents, setAttendedEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await fetch(`${API_URL}/events/events/`, {
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setEvents(data);
      }
    } catch (error) {
      console.error('Error loading events:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadAttendedEvents = async () => {
    try {
      const response = await fetch(`${API_URL}/events/attended-events/`, {
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAttendedEvents(data);
      }
    } catch (error) {
      console.error('Error loading attended events:', error);
    }
  };

  const markEventAttended = async (eventId) => {
    try {
      const response = await fetch(`${API_URL}/events/claim-event/${eventId}/`, {
        method: 'POST',
        headers: {
          'Authorization': `Token ${getToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        showNotification('Event marked as attended successfully!', 'success');
        loadEvents();
      } else {
        const data = await response.json();
        showNotification(data.error || 'Failed to mark event as attended', 'error');
      }
    } catch (error) {
      showNotification('Failed to mark event as attended', error);
    }
  };

  const showNotification = (message, type) => {
    setNotification({ show: true, message, type });
    setTimeout(() => {
      setNotification({ show: false, message: '', type: '' });
    }, 3000);
  };

  const getEventStatus = (event) => {
    const now = new Date();
    const start = new Date(event.start_time);
    const end = new Date(event.end_time);

    if (start > now) return { text: 'Upcoming', class: 'upcoming' };
    if (end < now) return { text: 'Past Event', class: 'past' };
    return { text: 'Ongoing', class: 'ongoing' };
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

  const filterEvents = (eventsList) => {
    const now = new Date();
    
    if (filter === 'all') return eventsList;
    
    return eventsList.filter(event => {
      const start = new Date(event.start_time);
      const end = new Date(event.end_time);
      
      if (filter === 'upcoming') return start > now;
      if (filter === 'ongoing') return start <= now && end >= now;
      if (filter === 'past') return end < now;
      return true;
    });
  };

  const navigateToEventsDashboard = () => {
    window.location.href = '/admin-dashboard/';
  };

  const EventCard = ({ event, showAttendButton = true }) => {
    const status = getEventStatus(event);
    const start = new Date(event.start_time);
    const end = new Date(event.end_time);

    return (
      <div className="bg-gray-900 rounded-lg overflow-hidden border border-blue-500/30 hover:border-blue-500 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20">
        <div className="bg-gradient-to-r from-blue-600 to-blue-500 p-4">
          <h3 className="text-white font-bold text-lg">{event.name}</h3>
        </div>
        
        <div className="p-4">
          <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold mb-3 ${
            status.class === 'ongoing' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' :
            status.class === 'upcoming' ? 'bg-gray-700 text-gray-300 border border-gray-600' :
            'bg-gray-800 text-gray-400 border border-gray-700'
          }`}>
            {status.text}
          </div>
          
          <p className="text-gray-400 mb-4">{event.description}</p>
          
          <div className="space-y-2 text-sm text-gray-300 mb-3">
            <p><span className="font-semibold text-white">Start:</span> {start.toLocaleString()}</p>
            <p><span className="font-semibold text-white">End:</span> {end.toLocaleString()}</p>
          </div>
          
          <span className="inline-block px-3 py-1 bg-blue-500/20 text-blue-400 rounded text-sm border border-blue-500/30">
            {getVisibilityLabel(event.visibility)}
          </span>
        </div>
        
        {showAttendButton && (
          <div className="border-t border-gray-800 p-4 flex justify-between items-center">
            {status.class === 'ongoing' || status.class === 'past' ? (
              <>
                <button
                  onClick={() => markEventAttended(event.id)}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70"
                >
                  Mark as Attended
                </button>
                {event.form_url && status.class === 'ongoing' && (
                  <a
                    href={event.form_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300 underline"
                  >
                    Registration Form
                  </a>
                )}
              </>
            ) : (
              <span className="text-gray-400 text-sm">Registration will open soon</span>
            )}
          </div>
        )}
      </div>
    );
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

      {/* Header */}
      <header className="bg-gradient-to-r from-gray-900 to-black border-b border-blue-500/30 shadow-lg shadow-blue-500/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center font-bold text-xl shadow-lg shadow-blue-500/50">
              F
            </div>
            <h1 className="text-2xl font-bold">FeedTrack</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={navigateToEventsDashboard}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70"
            >
              Events Dashboard
            </button>
            <button
              onClick={logout}
              className="bg-gray-900 hover:bg-gray-800 text-white px-6 py-2 rounded-lg border border-blue-500/30 hover:border-blue-500 transition-all duration-300"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* User Info Card */}
        <div className="bg-gradient-to-br from-gray-900 to-black rounded-lg border border-blue-500/30 p-6 mb-8 shadow-xl shadow-blue-500/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-2xl font-bold shadow-lg shadow-blue-500/50">
                {user?.name?.[0]?.toUpperCase() || 'T'}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Welcome back!</h2>
                <span className="inline-block mt-2 px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm border border-blue-500/50">
                  Teacher
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveTab('available')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all duration-300 ${
              activeTab === 'available'
                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                : 'bg-gray-900 text-gray-400 border border-gray-800 hover:border-blue-500/50'
            }`}
          >
            Available Events
          </button>
          <button
            onClick={() => {
              setActiveTab('attended');
              loadAttendedEvents();
            }}
            className={`px-6 py-3 rounded-lg font-semibold transition-all duration-300 ${
              activeTab === 'attended'
                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                : 'bg-gray-900 text-gray-400 border border-gray-800 hover:border-blue-500/50'
            }`}
          >
            Attended Events
          </button>
        </div>

        {/* Available Events Tab */}
        {activeTab === 'available' && (
          <div>
            {/* Filters */}
            <div className="flex gap-3 mb-6 flex-wrap">
              {['all', 'upcoming', 'ongoing', 'past'].map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 capitalize ${
                    filter === f
                      ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                      : 'bg-gray-900 text-gray-400 border border-gray-800 hover:border-blue-500/50'
                  }`}
                >
                  {f === 'all' ? 'All Events' : f}
                </button>
              ))}
            </div>

            {/* Events Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filterEvents(events).length > 0 ? (
                filterEvents(events).map(event => (
                  <EventCard key={event.id} event={event} />
                ))
              ) : (
                <div className="col-span-full text-center py-12 text-gray-400">
                  No events match the selected filter.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Attended Events Tab */}
        {activeTab === 'attended' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {attendedEvents.length > 0 ? (
              attendedEvents.map(event => (
                <EventCard key={event.id} event={event} showAttendButton={false} />
              ))
            ) : (
              <div className="col-span-full text-center py-12 text-gray-400">
                You have not attended any events yet.
              </div>
            )}
          </div>
        )}
      </div>

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

export default TeacherDashboard;