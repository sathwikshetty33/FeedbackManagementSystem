import { useState, useEffect } from 'react';
import useAuth from '../hooks/useAuth';

const TeacherDashboard = () => {
  const { user, getToken, API_URL } = useAuth();
  const [filter, setFilter] = useState('all');
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await fetch(`${API_URL}/events/teacher-events/`, {
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

    if (event.is_marked) {
      return { text: 'Attended', class: 'attended', showButton: false };
    }
    if (end < now) {
      return { text: 'Missed', class: 'missed', showButton: false };
    }
    if (start > now) {
      return { text: 'Coming Soon', class: 'upcoming', showButton: false };
    }
    return { text: 'Ongoing', class: 'ongoing', showButton: true };
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

      if (filter === 'upcoming') return start > now && !event.is_marked;
      if (filter === 'ongoing') return start <= now && end >= now && !event.is_marked;
      if (filter === 'attended') return event.is_marked;
      if (filter === 'missed') return end < now && !event.is_marked;

      return true;
    });
  };

  const EventCard = ({ event }) => {
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
            status.class === 'attended' ? 'bg-green-500/20 text-green-400 border border-green-500/50' :
            status.class === 'missed' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
            'bg-gray-800 text-gray-400 border border-gray-700'
          }`}>
            {status.text}
          </div>
          
          <p className="text-gray-400 mb-4">{event.description.substring(0, 100)}</p>
          
          <div className="space-y-2 text-sm text-gray-300 mb-3">
            <p><span className="font-semibold text-white">Start:</span> {start.toLocaleString()}</p>
            <p><span className="font-semibold text-white">End:</span> {end.toLocaleString()}</p>
            {event.created_by && (
              <p><span className="font-semibold text-white">Created by:</span> {event.created_by}</p>
            )}
          </div>
          
          <div className="flex gap-2 mb-3">
            <span className="inline-block px-3 py-1 bg-blue-500/20 text-blue-400 rounded text-sm border border-blue-500/30">
              {getVisibilityLabel(event.visibility)}
            </span>
          </div>
          
          {/* {event.worksheet_url && (
            <a
              href={event.worksheet_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm mb-3"
            >
              📄 View Worksheet
            </a>
          )} */}
        </div>
        
        <div className="border-t border-gray-800 p-4">
          {status.showButton ? (
            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
              <button
                onClick={() => markEventAttended(event.id)}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 w-full sm:w-auto"
              >
                Mark as Attended
              </button>
              {event.form_url && (
                <a
                  href={event.form_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 underline text-sm"
                >
                  Registration Form
                </a>
              )}
            </div>
          ) : status.class === 'attended' ? (
            <div className="flex items-center justify-between text-green-400 text-sm">
              <span className="font-semibold">✓ Attendance Confirmed</span>
            </div>
          ) : status.class === 'missed' ? (
            <span className="text-red-400 text-sm font-semibold">⚠ Event Ended - Attendance Missed</span>
          ) : (
            <span className="text-gray-400 text-sm">Event registration will open when it starts</span>
          )}
        </div>
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

  const filteredEvents = filterEvents(events);
  const attendedCount = events.filter(e => e.is_marked).length;
  const missedCount = events.filter(e => new Date(e.end_time) < new Date() && !e.is_marked).length;
  const ongoingCount = events.filter(e => {
    const now = new Date();
    const start = new Date(e.start_time);
    const end = new Date(e.end_time);
    return start <= now && end >= now && !e.is_marked;
  }).length;

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

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* User Info Card with Stats */}
        <div className="bg-gradient-to-br from-gray-900 to-black rounded-lg border border-blue-500/30 p-6 mb-8 shadow-xl shadow-blue-500/10">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-2xl font-bold shadow-lg shadow-blue-500/50">
                {user?.name?.[0]?.toUpperCase() || 'T'}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Welcome back!</h2>
                <span className="inline-block mt-2 px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm border border-blue-500/50">
                  {user?.username || 'Teacher'}
                </span>
              </div>
            </div>

            {/* Stats */}
            <div className="flex gap-4 flex-wrap">
              <div className="bg-black/50 border border-blue-500/30 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold text-blue-400">{attendedCount}</div>
                <div className="text-xs text-gray-400">Attended</div>
              </div>
              <div className="bg-black/50 border border-blue-500/30 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold text-blue-400">{ongoingCount}</div>
                <div className="text-xs text-gray-400">Ongoing</div>
              </div>
              <div className="bg-black/50 border border-red-500/30 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold text-red-400">{missedCount}</div>
                <div className="text-xs text-gray-400">Missed</div>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-white mb-3">Filter Events</h3>
          <div className="flex gap-3 flex-wrap">
            {[
              { key: 'all', label: 'All Events', count: events.length },
              { key: 'ongoing', label: 'Ongoing', count: ongoingCount },
              { key: 'upcoming', label: 'Coming Soon', count: events.filter(e => new Date(e.start_time) > new Date() && !e.is_marked).length },
              { key: 'attended', label: 'Attended', count: attendedCount },
              { key: 'missed', label: 'Missed', count: missedCount }
            ].map((f) => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                  filter === f.key
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                    : 'bg-gray-900 text-gray-400 border border-gray-800 hover:border-blue-500/50'
                }`}
              >
                {f.label} <span className="ml-1 opacity-70">({f.count})</span>
              </button>
            ))}
          </div>
        </div>

        {/* Events Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredEvents.length > 0 ? (
            filteredEvents.map(event => (
              <EventCard key={event.id} event={event} />
            ))
          ) : (
            <div className="col-span-full text-center py-12 bg-gray-900/50 rounded-lg border border-blue-500/20">
              <div className="text-6xl mb-4">📅</div>
              <p className="text-gray-400 text-lg">No events match the selected filter.</p>
            </div>
          )}
        </div>
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