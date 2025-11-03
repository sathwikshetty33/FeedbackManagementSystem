import { createContext, useState, useEffect } from 'react';
// import { useNavigate } from 'react-router-dom';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isDark, setIsDark] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Check if user is already logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      
      if (token) {
        try {
          // Optionally verify token with backend and get user data
          const response = await fetch(`${API_URL}/auth/verify/`, {
            headers: {
              'Authorization': `Token ${token}`
            }
          });
          
          if (response.ok) {
            const userData = await response.json();
            setUser(userData);
          } else {
            // Token invalid, clear it
            localStorage.removeItem('token');
          }
        } catch (error) {
          console.error('Auth check failed:', error);
          localStorage.removeItem('token');
        }
      }
      
      setLoading(false);
    };

    checkAuth();
  }, [API_URL]);

  // Apply theme to document
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  // Login function
  const login = (userData, token) => {
    localStorage.setItem('token', token);
    setUser(userData);
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const toggleTheme = () => {
    setIsDark(prev => !prev);
  };

  // Helper functions to get specific user data
  const getToken = () => localStorage.getItem('token');
  const getUserType = () => user?.user_type || '';
  const getUserId = () => user?.userId || user?.id || '';
  const getGithubAccessToken = () => user?.AccessToken || user?.access_token || '';

  return (
    <AuthContext.Provider value={{ 
      user,
      loading,
      API_URL,
      setUser,
      login,
      logout,
      getToken,
      getUserType,
      getUserId,
      getGithubAccessToken,
      isDark,
      toggleTheme,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;