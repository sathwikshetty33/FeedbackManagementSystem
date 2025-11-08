// import { useState, useEffect, useRef } from 'react';
// import useAuth from '../hooks/useAuth';

// const EventChat = () => {
//   const { getToken, API_URL } = useAuth();
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [sessionId, setSessionId] = useState('');
//   const [eventName, setEventName] = useState('');
//   const [initializing, setInitializing] = useState(true);
//   const messagesEndRef = useRef(null);

//   // Get URL parameters
//   const getUrlParams = () => {
//     const params = new URLSearchParams(window.location.search);
//     return {
//       eventId: params.get('event_id'),
//       dashboardUrl: params.get('dashboard_url') || '/events-dashboard',
//       eventName: params.get('event_name')
//     };
//   };

//   const { eventId, dashboardUrl, eventName: urlEventName } = getUrlParams();
//   console.log(eventName)
//   useEffect(() => {
//     const token = getToken();
//     if (!token) {
//       window.location.href = '/login';
//       return;
//     }

//     if (!eventId) {
//       window.location.href = '/events-dashboard';
//       return;
//     }

//     initializeChat();
//   }, []);

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   const initializeChat = async () => {
//     try {
//       // Call Django backend to initialize session with FastAPI
//       const response = await fetch(`${API_URL}/events/chat/${eventId}/`, {
//         headers: {
//           'Authorization': `Token ${getToken()}`
//         }
//       });

//       if (response.ok) {
//         const data = await response.json();
        
//         if (data.session_id) {
//           setSessionId(data.session_id);
//           setEventName(urlEventName || 'Event Chat');
          
//           // Add initial greeting message
//           setMessages([{
//             content: "👋 Hello! I'm here to help you with any questions or feedback about this event. What would you like to know?",
//             isUser: false,
//             timestamp: new Date()
//           }]);
//         } else {
//           throw new Error('Failed to get session ID from response');
//         }
//       } else {
//         const errorData = await response.json();
//         throw new Error(errorData.message || errorData.error || 'Failed to initialize chat session');
//       }
//     } catch (error) {
//       console.error('Error initializing chat:', error);
//       setMessages([{
//         content: "Sorry, there was an error initializing the chat. Please try again later.",
//         isUser: false,
//         timestamp: new Date()
//       }]);
//     } finally {
//       setInitializing(false);
//     }
//   };

//   const sendMessage = async () => {
//     const message = inputMessage.trim();
//     if (!message || loading) return;

//     // Add user message
//     const userMessage = {
//       content: message,
//       isUser: true,
//       timestamp: new Date()
//     };
//     setMessages(prev => [...prev, userMessage]);
//     setInputMessage('');
//     setLoading(true);

//     try {
//       const response = await fetch('http://localhost:8001/query', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//           session_id: sessionId,
//           question: message
//         })
//       });

//       const data = await response.json();
      
//       // Add assistant response
//       const assistantMessage = {
//         content: data.answer || data.error || "Sorry, I couldn't process your request.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error('Error sending message:', error);
//       const errorMessage = {
//         content: "Sorry, there was an error connecting to the server. Please try again.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       sendMessage();
//     }
//   };

//   const goBack = () => {
//     window.location.href = dashboardUrl;
//   };

//   if (initializing) {
//     return (
//       <div className="min-h-screen bg-black flex items-center justify-center">
//         <div className="text-center">
//           <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4 shadow-lg shadow-blue-500/50"></div>
//           <p className="text-white text-xl font-semibold">Initializing chat session...</p>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className="min-h-screen bg-black flex flex-col">
//       {/* Chat Container */}
//       <div className="flex-1 flex flex-col max-w-5xl w-full mx-auto px-4 py-8 gap-8">
//         {/* Header Section */}
//         <div className="flex items-center justify-between gap-4 flex-wrap">
//           <button
//             onClick={goBack}
//             className="bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-600 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/50 flex items-center gap-2"
//           >
//             <span>←</span>
//             <span>Back to Dashboard</span>
//           </button>
//           <h1 className="text-2xl sm:text-3xl font-bold text-white">
//             Feedback Chat Interface
//           </h1>
//           <div className="bg-gray-900 border border-blue-500/20 px-4 py-2 rounded-lg text-gray-300 text-sm shadow-lg shadow-blue-500/20">
//             Session: {sessionId.substring(0, 8)}...
//           </div>
//         </div>

//         {/* Messages Area */}
//         <div className="flex-1 bg-gray-900 rounded-2xl shadow-2xl shadow-blue-500/20 border border-blue-500/20 p-6 sm:p-8 overflow-y-auto min-h-[400px]">
//           <div className="space-y-6">
//             {messages.map((message, index) => (
//               <div
//                 key={index}
//                 className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
//               >
//                 <div
//                   className={`max-w-[80%] px-6 py-4 rounded-2xl ${
//                     message.isUser
//                       ? 'bg-blue-500 text-white rounded-br-lg shadow-lg shadow-blue-500/30'
//                       : 'bg-black text-gray-300 rounded-bl-lg border border-blue-500/30 shadow-md'
//                   }`}
//                 >
//                   <p className="text-base leading-relaxed whitespace-pre-wrap break-words">
//                     {message.content}
//                   </p>
//                 </div>
//               </div>
//             ))}
//             {loading && (
//               <div className="flex justify-start">
//                 <div className="bg-black text-gray-300 px-6 py-4 rounded-2xl rounded-bl-lg border border-blue-500/30 shadow-md">
//                   <div className="flex items-center gap-2">
//                     <span className="text-gray-400 italic">AI is thinking</span>
//                     <div className="flex gap-1">
//                       <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
//                       <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
//                       <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             )}
//             <div ref={messagesEndRef} />
//           </div>
//         </div>

//         {/* Input Area */}
//         <div className="bg-gray-900 rounded-2xl shadow-2xl shadow-blue-500/20 border border-blue-500/20 p-6">
//           <div className="flex gap-4 items-center flex-col sm:flex-row">
//             <input
//               type="text"
//               value={inputMessage}
//               onChange={(e) => setInputMessage(e.target.value)}
//               onKeyPress={handleKeyPress}
//               placeholder="Type your message here..."
//               disabled={loading}
//               className="flex-1 w-full sm:w-auto bg-black border-2 border-blue-500/30 focus:border-blue-500 px-6 py-4 rounded-xl outline-none transition-all duration-300 text-white placeholder-gray-400 focus:shadow-lg focus:shadow-blue-500/30"
//               autoFocus
//             />
//             <button
//               onClick={sendMessage}
//               disabled={loading || !inputMessage.trim()}
//               className="w-full sm:w-auto bg-blue-500 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-600 hover:shadow-xl hover:shadow-blue-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px]"
//             >
//               {loading ? 'Sending...' : 'Send'}
//             </button>
//           </div>
//         </div>
//       </div>

//       <style>{`
//         @keyframes fade-in {
//           from {
//             opacity: 0;
//             transform: translateY(20px);
//           }
//           to {
//             opacity: 1;
//             transform: translateY(0);
//           }
//         }
//         .animate-fade-in {
//           animation: fade-in 0.5s ease-out;
//         }
//       `}</style>
//     </div>
//   );
// };

// export default EventChat;

// import { useState, useEffect, useRef } from 'react';
// import useAuth from '../hooks/useAuth';

// const EventChat = () => {
//   const { getToken, API_URL } = useAuth();
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [sessionId, setSessionId] = useState('');
//   const [eventName, setEventName] = useState('');
//   const [initializing, setInitializing] = useState(true);
//   const messagesEndRef = useRef(null);

//   // Get URL parameters
//   const getUrlParams = () => {
//     const params = new URLSearchParams(window.location.search);
//     return {
//       eventId: params.get('event_id'),
//       dashboardUrl: params.get('dashboard_url') || '/events-dashboard',
//       eventName: params.get('event_name')
//     };
//   };

//   const { eventId, dashboardUrl, eventName: urlEventName } = getUrlParams();

//   useEffect(() => {
//     const token = getToken();
//     if (!token) {
//       window.location.href = '/login';
//       return;
//     }

//     if (!eventId) {
//       window.location.href = '/events-dashboard';
//       return;
//     }

//     initializeChat();
//   }, []);

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   const initializeChat = async () => {
//     try {
//       const response = await fetch(`${API_URL}/events/chat/${eventId}/`, {
//         headers: {
//           'Authorization': `Token ${getToken()}`
//         }
//       });

//       if (response.ok) {
//         const data = await response.json();
        
//         if (data.session_id) {
//           setSessionId(data.session_id);
//           setEventName(urlEventName || 'Event Chat');
          
//           setMessages([{
//             content: "👋 Hello! I'm here to help you with any questions or feedback about this event. What would you like to know?",
//             isUser: false,
//             timestamp: new Date()
//           }]);
//         } else {
//           throw new Error('Failed to get session ID from response');
//         }
//       } else {
//         const errorData = await response.json();
//         throw new Error(errorData.message || errorData.error || 'Failed to initialize chat session');
//       }
//     } catch (error) {
//       console.error('Error initializing chat:', error);
//       setMessages([{
//         content: "Sorry, there was an error initializing the chat. Please try again later.",
//         isUser: false,
//         timestamp: new Date()
//       }]);
//     } finally {
//       setInitializing(false);
//     }
//   };

//   const sendMessage = async () => {
//     const message = inputMessage.trim();
//     if (!message || loading) return;

//     const userMessage = {
//       content: message,
//       isUser: true,
//       timestamp: new Date()
//     };
//     setMessages(prev => [...prev, userMessage]);
//     setInputMessage('');
//     setLoading(true);

//     try {
//       const response = await fetch('http://localhost:8001/query', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//           session_id: sessionId,
//           question: message
//         })
//       });

//       const data = await response.json();
      
//       const assistantMessage = {
//         content: data.answer || data.error || "Sorry, I couldn't process your request.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error('Error sending message:', error);
//       const errorMessage = {
//         content: "Sorry, there was an error connecting to the server. Please try again.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       sendMessage();
//     }
//   };

//   if (initializing) {
//     return (
//       <div className="min-h-screen bg-black flex items-center justify-center">
//         <div className="text-center">
//           <div className="relative">
//             <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-500/30 border-t-blue-500 mx-auto mb-6 shadow-2xl shadow-blue-500/50"></div>
//             <div className="absolute inset-0 rounded-full bg-blue-500/10 blur-xl animate-pulse"></div>
//           </div>
//           <p className="text-white text-xl font-semibold mb-2">Initializing Chat</p>
//           <p className="text-gray-400 text-sm">Setting up your session...</p>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className="min-h-screen bg-black">
//       <div className="h-screen flex flex-col max-w-6xl mx-auto p-6">
        
//         {/* Chat Messages Container */}
//         <div className="flex-1 mb-6 relative">
//           <div className="absolute inset-0 bg-gradient-to-b from-gray-900/50 to-gray-900 rounded-3xl border border-blue-500/20 shadow-2xl shadow-blue-500/10 backdrop-blur-sm overflow-hidden">
            
//             {/* Session Info Bar */}
//             <div className="bg-black/40 border-b border-blue-500/20 px-6 py-3 flex items-center justify-between backdrop-blur-md">
//               <div className="flex items-center gap-3">
//                 <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse shadow-lg shadow-blue-500/50"></div>
//                 <span className="text-gray-300 text-sm font-medium">Active Session</span>
//               </div>
//               <div className="text-gray-400 text-xs font-mono">
//                 ID: {sessionId.substring(0, 12)}
//               </div>
//             </div>

//             {/* Messages Area */}
//             <div className="h-[calc(100%-3.5rem)] overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent">
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} animate-slide-in`}
//                 >
//                   {!message.isUser && (
//                     <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30 flex-shrink-0">
//                       <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//                       </svg>
//                     </div>
//                   )}
                  
//                   <div
//                     className={`max-w-[70%] ${
//                       message.isUser
//                         ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-xl shadow-blue-500/30 rounded-3xl rounded-br-md'
//                         : 'bg-gray-900 text-gray-200 border border-blue-500/20 shadow-lg rounded-3xl rounded-bl-md'
//                     } px-6 py-4 transition-all duration-300 hover:scale-[1.02]`}
//                   >
//                     <p className="text-base leading-relaxed whitespace-pre-wrap break-words">
//                       {message.content}
//                     </p>
//                     <p className={`text-xs mt-2 ${message.isUser ? 'text-blue-100' : 'text-gray-500'}`}>
//                       {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
//                     </p>
//                   </div>

//                   {message.isUser && (
//                     <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center ml-3 shadow-lg flex-shrink-0">
//                       <svg className="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
//                       </svg>
//                     </div>
//                   )}
//                 </div>
//               ))}
              
//               {loading && (
//                 <div className="flex justify-start animate-slide-in">
//                   <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30">
//                     <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//                     </svg>
//                   </div>
//                   <div className="bg-gray-900 border border-blue-500/20 px-6 py-4 rounded-3xl rounded-bl-md shadow-lg">
//                     <div className="flex items-center gap-3">
//                       <div className="flex gap-1.5">
//                         <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0s' }}></div>
//                         <div className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce shadow-lg shadow-blue-400/50" style={{ animationDelay: '0.15s' }}></div>
//                         <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0.3s' }}></div>
//                       </div>
//                       <span className="text-gray-400 text-sm">Thinking...</span>
//                     </div>
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </div>
//           </div>
//         </div>

//         {/* Input Area */}
//         <div className="relative">
//           <div className="bg-gradient-to-br from-gray-900 to-black rounded-2xl border border-blue-500/30 shadow-2xl shadow-blue-500/20 p-4 backdrop-blur-sm">
//             <div className="flex gap-3 items-end">
//               <div className="flex-1 relative">
//                 <textarea
//                   value={inputMessage}
//                   onChange={(e) => setInputMessage(e.target.value)}
//                   onKeyPress={handleKeyPress}
//                   placeholder="Ask me anything about this event..."
//                   disabled={loading}
//                   rows={1}
//                   className="w-full bg-black/50 border border-blue-500/30 focus:border-blue-500 px-5 py-3.5 rounded-xl outline-none transition-all duration-300 text-white placeholder-gray-500 focus:shadow-lg focus:shadow-blue-500/20 resize-none scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent"
//                   style={{ minHeight: '52px', maxHeight: '120px' }}
//                   onInput={(e) => {
//                     e.target.style.height = 'auto';
//                     e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
//                   }}
//                 />
//               </div>
//               <button
//                 onClick={sendMessage}
//                 disabled={loading || !inputMessage.trim()}
//                 className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-8 py-3.5 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 hover:shadow-xl hover:shadow-blue-500/40 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none flex items-center gap-2 group"
//               >
//                 {loading ? (
//                   <>
//                     <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
//                     <span>Sending</span>
//                   </>
//                 ) : (
//                   <>
//                     <span>Send</span>
//                     <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
//                     </svg>
//                   </>
//                 )}
//               </button>
//             </div>
//           </div>
//         </div>
//       </div>

//       <style>{`
//         @keyframes slide-in {
//           from {
//             opacity: 0;
//             transform: translateY(10px);
//           }
//           to {
//             opacity: 1;
//             transform: translateY(0);
//           }
//         }
//         .animate-slide-in {
//           animation: slide-in 0.4s ease-out;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar {
//           width: 6px;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-track {
//           background: transparent;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-thumb {
//           background: rgba(59, 130, 246, 0.2);
//           border-radius: 3px;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-thumb:hover {
//           background: rgba(59, 130, 246, 0.3);
//         }
//       `}</style>
//     </div>
//   );
// };

// export default EventChat;


// import { useState, useEffect, useRef } from 'react';
// import useAuth from '../hooks/useAuth';

// const EventChat = () => {
//   const { getToken, API_URL } = useAuth();
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [sessionId, setSessionId] = useState('');
//   const [eventName, setEventName] = useState('');
//   const [initializing, setInitializing] = useState(true);
//   const [isListening, setIsListening] = useState(false);
//   const [recognition, setRecognition] = useState(null);
//   const messagesEndRef = useRef(null);
//   const messagesContainerRef = useRef(null);

//   // Get URL parameters
//   const getUrlParams = () => {
//     const params = new URLSearchParams(window.location.search);
//     return {
//       eventId: params.get('event_id'),
//       dashboardUrl: params.get('dashboard_url') || '/events-dashboard',
//       eventName: params.get('event_name')
//     };
//   };

//   const { eventId, eventName: urlEventName } = getUrlParams();

//   // Initialize speech recognition
//   useEffect(() => {
//     if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
//       const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
//       const recognitionInstance = new SpeechRecognition();
//       recognitionInstance.continuous = false;
//       recognitionInstance.interimResults = false;
//       recognitionInstance.lang = 'en-US';

//       recognitionInstance.onresult = (event) => {
//         const transcript = event.results[0][0].transcript;
//         setInputMessage(prev => prev + (prev ? ' ' : '') + transcript);
//       };

//       recognitionInstance.onend = () => {
//         setIsListening(false);
//       };

//       recognitionInstance.onerror = (event) => {
//         console.error('Speech recognition error:', event.error);
//         setIsListening(false);
//       };

//       setRecognition(recognitionInstance);
//     }
//   }, []);

//   useEffect(() => {
//     const token = getToken();
//     if (!token) {
//       window.location.href = '/login';
//       return;
//     }

//     if (!eventId) {
//       window.location.href = '/events-dashboard';
//       return;
//     }

//     initializeChat();
//   }, []);

//   useEffect(() => {
//     // Only auto-scroll when new messages are added (not on initial load)
//     if (messages.length > 1 && messagesContainerRef.current) {
//       const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
//       const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
      
//       if (isNearBottom) {
//         scrollToBottom();
//       }
//     }
//   }, [messages]);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   const initializeChat = async () => {
//     try {
//       const response = await fetch(`${API_URL}/events/chat/${eventId}/`, {
//         headers: {
//           'Authorization': `Token ${getToken()}`
//         }
//       });

//       if (response.ok) {
//         const data = await response.json();
        
//         if (data.session_id) {
//           setSessionId(data.session_id);
//           setEventName(urlEventName || 'Event Chat');
          
//           setMessages([{
//             content: "👋 Hello! I'm here to help you with any questions or feedback about this event. What would you like to know?",
//             isUser: false,
//             timestamp: new Date()
//           }]);
//         } else {
//           throw new Error('Failed to get session ID from response');
//         }
//       } else {
//         const errorData = await response.json();
//         throw new Error(errorData.message || errorData.error || 'Failed to initialize chat session');
//       }
//     } catch (error) {
//       console.error('Error initializing chat:', error);
//       setMessages([{
//         content: "Sorry, there was an error initializing the chat. Please try again later.",
//         isUser: false,
//         timestamp: new Date()
//       }]);
//     } finally {
//       setInitializing(false);
//     }
//   };

//   const sendMessage = async () => {
//     const message = inputMessage.trim();
//     if (!message || loading) return;

//     const userMessage = {
//       content: message,
//       isUser: true,
//       timestamp: new Date()
//     };
//     setMessages(prev => [...prev, userMessage]);
//     setInputMessage('');
//     setLoading(true);

//     try {
//       const response = await fetch('http://localhost:8001/query', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//           session_id: sessionId,
//           question: message
//         })
//       });

//       const data = await response.json();
      
//       const assistantMessage = {
//         content: data.answer || data.error || "Sorry, I couldn't process your request.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error('Error sending message:', error);
//       const errorMessage = {
//         content: "Sorry, there was an error connecting to the server. Please try again.",
//         isUser: false,
//         timestamp: new Date()
//       };
//       setMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const toggleSpeechRecognition = () => {
//     if (!recognition) {
//       alert('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
//       return;
//     }

//     if (isListening) {
//       recognition.stop();
//       setIsListening(false);
//     } else {
//       recognition.start();
//       setIsListening(true);
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       e.stopPropagation();
//       sendMessage();
//     }
//   };

//   if (initializing) {
//     return (
//       <div className="min-h-screen bg-black flex items-center justify-center">
//         <div className="text-center">
//           <div className="relative">
//             <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-500/30 border-t-blue-500 mx-auto mb-6 shadow-2xl shadow-blue-500/50"></div>
//             <div className="absolute inset-0 rounded-full bg-blue-500/10 blur-xl animate-pulse"></div>
//           </div>
//           <p className="text-white text-xl font-semibold mb-2">Initializing Chat</p>
//           <p className="text-gray-400 text-sm">Setting up your session...</p>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className="min-h-screen bg-black overflow-hidden">
//       <div className="h-screen flex flex-col max-w-4xl mx-auto p-4 sm:p-6 overflow-hidden">
        
//         {/* Main Chat Container */}
//         <div className="flex-1 relative overflow-hidden min-h-0">
//           <div className="absolute inset-0 bg-gradient-to-b from-gray-900/50 to-gray-900 rounded-3xl border border-blue-500/20 shadow-2xl shadow-blue-500/10 backdrop-blur-sm flex flex-col overflow-hidden">
            
//             {/* Session Info Bar */}
//             <div className="bg-black/40 border-b border-blue-500/20 px-6 py-3 flex items-center justify-between backdrop-blur-md flex-shrink-0">
//               <div className="flex items-center gap-3">
//                 <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse shadow-lg shadow-blue-500/50"></div>
//                 <span className="text-gray-300 text-sm font-medium">Active Session</span>
//               </div>
//               <div className="text-gray-400 text-xs font-mono">
//                 ID: {sessionId.substring(0, 12)}
//               </div>
//             </div>

//             {/* Messages Area */}
//             <div 
//               ref={messagesContainerRef}
//               className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6 scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent"
//             >
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} animate-slide-in`}
//                 >
//                   {!message.isUser && (
//                     <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30 flex-shrink-0">
//                       <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//                       </svg>
//                     </div>
//                   )}
                  
//                   <div
//                     className={`max-w-[85%] ${
//                       message.isUser
//                         ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-xl shadow-blue-500/30 rounded-3xl rounded-br-md'
//                         : 'bg-gray-900 text-gray-200 border border-blue-500/20 shadow-lg rounded-3xl rounded-bl-md'
//                     } px-4 sm:px-6 py-3 sm:py-4 transition-all duration-300 hover:scale-[1.02]`}
//                   >
//                     <p className="text-base leading-relaxed whitespace-pre-wrap break-words">
//                       {message.content}
//                     </p>
//                     <p className={`text-xs mt-2 ${message.isUser ? 'text-blue-100' : 'text-gray-500'}`}>
//                       {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
//                     </p>
//                   </div>

//                   {message.isUser && (
//                     <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center ml-3 shadow-lg flex-shrink-0">
//                       <svg className="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
//                       </svg>
//                     </div>
//                   )}
//                 </div>
//               ))}
              
//               {loading && (
//                 <div className="flex justify-start animate-slide-in">
//                   <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30">
//                     <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//                     </svg>
//                   </div>
//                   <div className="bg-gray-900 border border-blue-500/20 px-6 py-4 rounded-3xl rounded-bl-md shadow-lg">
//                     <div className="flex items-center gap-3">
//                       <div className="flex gap-1.5">
//                         <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0s' }}></div>
//                         <div className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce shadow-lg shadow-blue-400/50" style={{ animationDelay: '0.15s' }}></div>
//                         <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0.3s' }}></div>
//                       </div>
//                       <span className="text-gray-400 text-sm">Thinking...</span>
//                     </div>
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </div>

//             {/* Input Area Inside Chat Box */}
//             <div className="border-t border-blue-500/20 bg-black/20 backdrop-blur-md p-4 flex-shrink-0">
//               <div className="flex gap-3 items-end">
//                 <div className="flex-1 relative">
//                   <textarea
//                     value={inputMessage}
//                     onChange={(e) => setInputMessage(e.target.value)}
//                     onKeyPress={handleKeyPress}
//                     placeholder="Ask me anything about this event..."
//                     disabled={loading}
//                     rows={1}
//                     className="w-full bg-gray-900/70 border border-blue-500/30 focus:border-blue-500 px-5 py-3.5 pr-14 rounded-xl outline-none transition-all duration-300 text-white placeholder-gray-500 focus:shadow-lg focus:shadow-blue-500/20 resize-none scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent"
//                     style={{ minHeight: '52px', maxHeight: '120px' }}
//                     onInput={(e) => {
//                       e.target.style.height = 'auto';
//                       e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
//                     }}
//                   />
//                   <button
//                     onClick={toggleSpeechRecognition}
//                     disabled={loading}
//                     className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-all duration-300 ${
//                       isListening 
//                         ? 'bg-red-500 text-white shadow-lg shadow-red-500/50 animate-pulse' 
//                         : 'bg-gray-800 text-gray-400 hover:bg-blue-500 hover:text-white hover:shadow-lg hover:shadow-blue-500/30'
//                     }`}
//                     title={isListening ? 'Stop recording' : 'Start voice input'}
//                   >
//                     <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
//                     </svg>
//                   </button>
//                 </div>
//                 <button
//                   onClick={sendMessage}
//                   disabled={loading || !inputMessage.trim()}
//                   className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-8 py-3.5 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 hover:shadow-xl hover:shadow-blue-500/40 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none flex items-center gap-2 group"
//                 >
//                   {loading ? (
//                     <>
//                       <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
//                       <span>Sending</span>
//                     </>
//                   ) : (
//                     <>
//                       <span>Send</span>
//                       <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
//                       </svg>
//                     </>
//                   )}
//                 </button>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>

//       <style>{`
//         @keyframes slide-in {
//           from {
//             opacity: 0;
//             transform: translateY(10px);
//           }
//           to {
//             opacity: 1;
//             transform: translateY(0);
//           }
//         }
//         .animate-slide-in {
//           animation: slide-in 0.4s ease-out;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar {
//           width: 6px;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-track {
//           background: transparent;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-thumb {
//           background: rgba(59, 130, 246, 0.2);
//           border-radius: 3px;
//         }
        
//         .scrollbar-thin::-webkit-scrollbar-thumb:hover {
//           background: rgba(59, 130, 246, 0.3);
//         }
//       `}</style>
//     </div>
//   );
// };

// export default EventChat;
import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import useAuth from '../hooks/useAuth';

const EventChat = () => {
  const { getToken, API_URL } = useAuth();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [eventName, setEventName] = useState('');
  const [initializing, setInitializing] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Get URL parameters
  const getUrlParams = () => {
    const params = new URLSearchParams(window.location.search);
    return {
      eventId: params.get('event_id'),
      dashboardUrl: params.get('dashboard_url') || '/events-dashboard',
      eventName: params.get('event_name')
    };
  };

  const { eventId, dashboardUrl, eventName: urlEventName } = getUrlParams();

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = false;
      recognitionInstance.lang = 'en-US';

      recognitionInstance.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(prev => prev + (prev ? ' ' : '') + transcript);
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
      };

      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      setRecognition(recognitionInstance);
    }
  }, []);

  useEffect(() => {
    const token = getToken();
    if (!token) {
      window.location.href = '/login';
      return;
    }

    if (!eventId) {
      window.location.href = '/events-dashboard';
      return;
    }

    initializeChat();
  }, []);

  // ✅ Fixed Auto-scroll Behavior
  useLayoutEffect(() => {
    if (!messagesContainerRef.current) return;
    const container = messagesContainerRef.current;

    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 200;

    if (isNearBottom || messages.length <= 1) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: messages.length <= 1 ? 'auto' : 'smooth'
      });
    }
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTo({
        top: messagesContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  const initializeChat = async () => {
    try {
      const response = await fetch(`${API_URL}/events/chat/${eventId}/`, {
        headers: {
          'Authorization': `Token ${getToken()}`
        }
      });

      if (response.ok) {
        const data = await response.json();

        if (data.session_id) {
          setSessionId(data.session_id);
          setEventName(urlEventName || 'Event Chat');

          setMessages([{
            content: "👋 Hello! I'm here to help you with any questions or feedback about this event. What would you like to know?",
            isUser: false,
            timestamp: new Date()
          }]);
        } else {
          throw new Error('Failed to get session ID from response');
        }
      } else {
        const errorData = await response.json();
        throw new Error(errorData.message || errorData.error || 'Failed to initialize chat session');
      }
    } catch (error) {
      console.error('Error initializing chat:', error);
      setMessages([{
        content: "Sorry, there was an error initializing the chat. Please try again later.",
        isUser: false,
        timestamp: new Date()
      }]);
    } finally {
      setInitializing(false);
    }
  };

  const sendMessage = async () => {
    const message = inputMessage.trim();
    if (!message || loading) return;

    const userMessage = {
      content: message,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8001/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId,
          question: message
        })
      });

      const data = await response.json();

      const assistantMessage = {
        content: data.answer || data.error || "Sorry, I couldn't process your request.",
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        content: "Sorry, there was an error connecting to the server. Please try again.",
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const toggleSpeechRecognition = () => {
    if (!recognition) {
      alert('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      recognition.start();
      setIsListening(true);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();
      sendMessage();
    }
  };

  if (initializing) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-500/30 border-t-blue-500 mx-auto mb-6 shadow-2xl shadow-blue-500/50"></div>
            <div className="absolute inset-0 rounded-full bg-blue-500/10 blur-xl animate-pulse"></div>
          </div>
          <p className="text-white text-xl font-semibold mb-2">Initializing Chat</p>
          <p className="text-gray-400 text-sm">Setting up your session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen max-h-screen bg-black overflow-hidden fixed inset-0">
      <div className="h-full flex flex-col max-w-4xl mx-auto p-4 sm:p-6 overflow-hidden">
        
        {/* Main Chat Container */}
        <div className="flex-1 relative overflow-hidden min-h-0">
          <div className="absolute inset-0 bg-gradient-to-b from-gray-900/50 to-gray-900 rounded-3xl border border-blue-500/20 shadow-2xl shadow-blue-500/10 backdrop-blur-sm flex flex-col overflow-hidden">
            
            {/* Session Info Bar */}
            <div className="bg-black/40 border-b border-blue-500/20 px-6 py-3 flex items-center justify-between backdrop-blur-md flex-shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse shadow-lg shadow-blue-500/50"></div>
                <span className="text-gray-300 text-sm font-medium">Active Session</span>
              </div>
              <div className="text-gray-400 text-xs font-mono">
                ID: {sessionId.substring(0, 12)}
              </div>
            </div>

            {/* Messages Area */}
            <div 
              ref={messagesContainerRef}
              className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6 scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent"
            >
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} animate-slide-in`}
                >
                  {!message.isUser && (
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30 flex-shrink-0">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                  )}
                  
                  <div
                    className={`max-w-[85%] ${
                      message.isUser
                        ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-xl shadow-blue-500/30 rounded-3xl rounded-br-md'
                        : 'bg-gray-900 text-gray-200 border border-blue-500/20 shadow-lg rounded-3xl rounded-bl-md'
                    } px-4 sm:px-6 py-3 sm:py-4 transition-all duration-300 hover:scale-[1.02]`}
                  >
                    <p className="text-base leading-relaxed whitespace-pre-wrap break-words">
                      {message.content}
                    </p>
                    <p className={`text-xs mt-2 ${message.isUser ? 'text-blue-100' : 'text-gray-500'}`}>
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>

                  {message.isUser && (
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center ml-3 shadow-lg flex-shrink-0">
                      <svg className="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                  )}
                </div>
              ))}
              
              {loading && (
                <div className="flex justify-start animate-slide-in">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/30">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div className="bg-gray-900 border border-blue-500/20 px-6 py-4 rounded-3xl rounded-bl-md shadow-lg">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0s' }}></div>
                        <div className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce shadow-lg shadow-blue-400/50" style={{ animationDelay: '0.15s' }}></div>
                        <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0.3s' }}></div>
                      </div>
                      <span className="text-gray-400 text-sm">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area Inside Chat Box */}
            <div className="border-t border-blue-500/20 bg-black/20 backdrop-blur-md p-4 flex-shrink-0">
              <div className="flex gap-3 items-end">
                <div className="flex-1 relative">
                  <textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me anything about this event..."
                    disabled={loading}
                    rows={1}
                    className="w-full bg-gray-900/70 border border-blue-500/30 focus:border-blue-500 px-5 py-3.5 pr-14 rounded-xl outline-none transition-all duration-300 text-white placeholder-gray-500 focus:shadow-lg focus:shadow-blue-500/20 resize-none scrollbar-thin scrollbar-thumb-blue-500/20 scrollbar-track-transparent"
                    style={{ minHeight: '52px', maxHeight: '120px' }}
                    onInput={(e) => {
                      e.target.style.height = 'auto';
                      e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                    }}
                  />
                  <button
                    onClick={toggleSpeechRecognition}
                    disabled={loading}
                    className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-all duration-300 ${
                      isListening 
                        ? 'bg-red-500 text-white shadow-lg shadow-red-500/50 animate-pulse' 
                        : 'bg-gray-800 text-gray-400 hover:bg-blue-500 hover:text-white hover:shadow-lg hover:shadow-blue-500/30'
                    }`}
                    title={isListening ? 'Stop recording' : 'Start voice input'}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </button>
                </div>
                <button
                  onClick={sendMessage}
                  disabled={loading || !inputMessage.trim()}
                  className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-8 py-3.5 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 hover:shadow-xl hover:shadow-blue-500/40 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none flex items-center gap-2 group"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Sending</span>
                    </>
                  ) : (
                    <>
                      <span>Send</span>
                      <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                      </svg>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slide-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slide-in {
          animation: slide-in 0.4s ease-out;
        }
        
        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: rgba(59, 130, 246, 0.2);
          border-radius: 3px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: rgba(59, 130, 246, 0.3);
        }
      `}</style>
    </div>
  );
};

export default EventChat;
