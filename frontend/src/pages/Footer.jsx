
const Footer = () => {
  return (
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
  )
}

export default Footer;