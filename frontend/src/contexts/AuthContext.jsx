import { createContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const AuthContext = createContext();


export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isDark, setIsDark] = useState(() => {
    // Check localStorage first, then system preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  
  const navigate = useNavigate();
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

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

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // Verify token and get user data
      fetchUserData(token);
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUserData = async (token) => {
    try {
      const response = await fetch(`${API_URL}/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const userData = await response.json();
        console.log('Fetched user data:', userData);
        setUser(userData);
      } else {
        localStorage.removeItem('token');
      }
    } catch (error) {
      console.error('Failed to fetch user data:', error);
      localStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  };

  const login = async (username, password) => {
    const response = await fetch(`${API_URL}/api/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Login failed');
    }

    const data = await response.json();
    localStorage.setItem('token', data.token);
    await fetchUserData(data.token);
    navigate('/dashboard');
  };

  const signup = async (username, email, password) => {
    const response = await fetch(`${API_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Signup failed');
    }

    const data = await response.json();
    localStorage.setItem('token', data.token);
    await fetchUserData(data.token);
    navigate('/dashboard');
  };

  const githubAuth = async (code) => {
    const response = await fetch(`${API_URL}/api/auth/github`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      throw new Error('GitHub authentication failed');
    }

    const data = await response.json();
    localStorage.setItem('token', data.token);
    await fetchUserData(data.token);
    navigate('/dashboard');
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
    navigate('/');
  };

  const toggleTheme = () => {
    setIsDark(prev => !prev);
  };

  // Helper functions to get specific user data
  const getToken = () => localStorage.getItem('token');
  const getGithubUrl = () => user?.GithubUrl || user?.github_url || '';
  const getGithubAccessToken = () => user?.AccessToken || user?.access_token || '';
  const getGithubUsername = () => {
    const url = getGithubUrl();
    if (!url) return '';
    return url.split('github.com/')[1]?.replace('/', '') || '';
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      loading, 
      login, 
      signup, 
      githubAuth, 
      logout,
      API_URL,
      getToken,
      getGithubUrl,
      getGithubAccessToken,
      getGithubUsername,
      isDark,
      toggleTheme,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;