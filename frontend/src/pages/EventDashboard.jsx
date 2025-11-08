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
  const [formModal, setFormModal] = useState({ 
    show: false, 
    isEditMode: false, 
    eventId: null,
    loading: false
  });
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    start_time: '',
    end_time: '',
    visibility: 'anyone',
    form_url: '',
    worksheet_url: ''
  });

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
      'anyone': 'Anyone', 'teachers': 'Teachers Only', 'students': 'Students'
    };
    return map[visibility] || visibility;
  };

  const closeModal = () => {
    setInsightsModal({ show: false, eventId: null, eventName: '', loading: false });
  };

  // Event Form Functions
  const openCreateForm = () => {
    setFormData({
      name: '',
      description: '',
      start_time: '',
      end_time: '',
      visibility: 'anyone',
      form_url: '',
      worksheet_url: ''
    });
    setFormModal({ show: true, isEditMode: false, eventId: null, loading: false });
  };

  const openEditForm = async (eventId) => {
    setFormModal({ show: true, isEditMode: true, eventId, loading: true });

    try {
      const response = await fetch(`${API_URL}/events/events/${eventId}/`, {
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        const event = await response.json();
        
        const formatDateForInput = (dateString) => {
          const date = new Date(dateString);
          const year = date.getFullYear();
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const day = String(date.getDate()).padStart(2, '0');
          const hours = String(date.getHours()).padStart(2, '0');
          const minutes = String(date.getMinutes()).padStart(2, '0');
          return `${year}-${month}-${day}T${hours}:${minutes}`;
        };

        setFormData({
          name: event.name,
          description: event.description,
          start_time: formatDateForInput(event.start_time),
          end_time: formatDateForInput(event.end_time),
          visibility: event.visibility,
          form_url: event.form_url,
          worksheet_url: event.worksheet_url
        });
        setFormModal(prev => ({ ...prev, loading: false }));
      } else {
        throw new Error('Failed to fetch event details');
      }
    } catch (error) {
      console.error('Error:', error);
      showNotification('Error loading event details', 'error');
      closeFormModal();
    }
  };

  const closeFormModal = () => {
    setFormModal({ show: false, isEditMode: false, eventId: null, loading: false });
  };

  const handleFormInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.description || !formData.start_time || 
        !formData.end_time || !formData.visibility || !formData.form_url || 
        !formData.worksheet_url) {
      showNotification('Please fill out all fields', 'error');
      return;
    }

    setFormModal(prev => ({ ...prev, loading: true }));

    try {
      const eventData = {
        ...formData,
        start_time: new Date(formData.start_time).toISOString(),
        end_time: new Date(formData.end_time).toISOString()
      };

      const method = formModal.isEditMode ? 'PUT' : 'POST';
      const endpoint = formModal.isEditMode 
        ? `${API_URL}/events/events/${formModal.eventId}/` 
        : `${API_URL}/events/create-events/`;

      const response = await fetch(endpoint, {
        method: method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Token ${getToken()}`
        },
        body: JSON.stringify(eventData)
      });

      if (response.ok) {
        showNotification(
          formModal.isEditMode ? 'Event updated successfully' : 'Event created successfully',
          'success'
        );
        closeFormModal();
        loadEvents();
      } else {
        throw new Error('Failed to save event');
      }
    } catch (error) {
      console.error('Error:', error);
      showNotification('Error saving event', 'error');
    } finally {
      setFormModal(prev => ({ ...prev, loading: false }));
    }
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

      {/* Event Form Modal */}
      {formModal.show && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-gray-900 rounded-lg border border-blue-500/30 shadow-2xl shadow-blue-500/20 max-w-2xl w-full my-8">
            <div className="flex justify-between items-center p-6 border-b border-gray-800">
              <h3 className="text-2xl font-bold text-white">
                {formModal.isEditMode ? 'Edit Event' : 'Create New Event'}
              </h3>
              <button
                onClick={closeFormModal}
                className="text-gray-400 hover:text-white text-2xl transition-colors"
              >
                ×
              </button>
            </div>
            
            <div className="p-6 max-h-[70vh] overflow-y-auto">
              {formModal.loading && formModal.isEditMode ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                  <p className="text-gray-300">Loading event details...</p>
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Event Name */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Event Name</label>
                    <input
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors"
                      required
                    />
                  </div>

                  {/* Description */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Description</label>
                    <textarea
                      name="description"
                      value={formData.description}
                      onChange={handleFormInputChange}
                      rows="4"
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors resize-vertical"
                      required
                    />
                  </div>

                  {/* Start Time */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Start Date & Time</label>
                    <input
                      type="datetime-local"
                      name="start_time"
                      value={formData.start_time}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors [color-scheme:dark]"
                      required
                    />
                  </div>

                  {/* End Time */}
                  <div>
                    <label className="block text-white font-semibold mb-2">End Date & Time</label>
                    <input
                      type="datetime-local"
                      name="end_time"
                      value={formData.end_time}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors [color-scheme:dark]"
                      required
                    />
                  </div>

                  {/* Visibility */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Visibility</label>
                    <select
                      name="visibility"
                      value={formData.visibility}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors"
                      required
                    >
                      <option value="anyone">Anyone</option>
                      <option value="students">Students</option>
                      <option value="teachers">Teachers Only</option>
                      <option value="1">Semester 1</option>
                      <option value="2">Semester 2</option>
                      <option value="3">Semester 3</option>
                      <option value="4">Semester 4</option>
                      <option value="5">Semester 5</option>
                      <option value="6">Semester 6</option>
                      <option value="7">Semester 7</option>
                      <option value="8">Semester 8</option>
                    </select>
                  </div>

                  {/* Registration Form URL */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Registration Form URL</label>
                    <input
                      type="url"
                      name="form_url"
                      value={formData.form_url}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors"
                      placeholder="https://example.com/form"
                      required
                    />
                  </div>

                  {/* Worksheet URL */}
                  <div>
                    <label className="block text-white font-semibold mb-2">Worksheet URL</label>
                    <input
                      type="url"
                      name="worksheet_url"
                      value={formData.worksheet_url}
                      onChange={handleFormInputChange}
                      className="w-full px-4 py-3 bg-black border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none transition-colors"
                      placeholder="https://example.com/worksheet"
                      required
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="p-6 border-t border-gray-800 flex justify-end gap-3">
              <button
                onClick={closeFormModal}
                className="bg-red-500/20 text-red-400 border border-red-500/50 px-6 py-2.5 rounded-lg font-semibold hover:bg-red-500/30 transition-all duration-300"
              >
                Cancel
              </button>
              <button
                onClick={handleFormSubmit}
                disabled={formModal.loading}
                className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2.5 rounded-lg font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {formModal.loading ? 'Saving...' : formModal.isEditMode ? 'Update Event' : 'Create Event'}
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
            onClick={openCreateForm}
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
              onClick={openCreateForm}
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
                    <th className="px-6 py-4 text-left text-white font-semibold">Form URL</th>
                    <th className="px-6 py-4 text-left text-white font-semibold">Worksheet URL</th>
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
                        <a
                          href={event.form_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-green-400 hover:text-green-300 underline transition-colors"
                        >
                          Open Form
                        </a>
                      </td>
                      <td className="px-6 py-4">
                        <a
                          href={event.worksheet_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-purple-400 hover:text-purple-300 underline transition-colors"
                        >
                          Open Worksheet
                        </a>
                      </td>
                      <td className="px-6 py-4 text-right">
  <div className="relative group inline-block">
    <button className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-white rounded-md text-sm transition">
      ⋮
    </button>
    {/* Remove mt-2 gap and add pt-2 padding inside to maintain visual spacing */}
    <div className="absolute right-0 pt-2 hidden group-hover:block z-10">
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-lg min-w-[150px]">
        <button
          onClick={() => generateInsights(event.id, event.name)}
          className="block w-full text-left px-4 py-2 hover:bg-blue-600/20 text-blue-400 text-sm first:rounded-t-lg"
        >
          📊 Insights
        </button>
        <button
          onClick={() => openChat(event.id, event.name)}
          className="block w-full text-left px-4 py-2 hover:bg-gray-600/20 text-gray-300 text-sm"
        >
          💬 Chat
        </button>
        <button
          onClick={() => openEditForm(event.id)}
          className="block w-full text-left px-4 py-2 hover:bg-yellow-600/20 text-yellow-400 text-sm"
        >
          ✏️ Edit
        </button>
        <button
          onClick={() => deleteEvent(event.id)}
          className="block w-full text-left px-4 py-2 hover:bg-red-600/20 text-red-400 text-sm last:rounded-b-lg"
        >
          🗑️ Delete
        </button>
      </div>
    </div>
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