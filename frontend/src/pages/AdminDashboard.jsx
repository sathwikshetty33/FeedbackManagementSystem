
 const AdminDashboard = () => {
  const team = [
    { name: 'Alex Johnson', role: 'CEO & Founder', avatar: '👨‍💼' },
    { name: 'Sarah Williams', role: 'CTO', avatar: '👩‍💻' },
    { name: 'Michael Chen', role: 'Lead Designer', avatar: '👨‍🎨' },
    { name: 'Emily Davis', role: 'Product Manager', avatar: '👩‍💼' }
  ]

  const features = [
    {
      title: 'Modern Architecture',
      description: 'Built with React, Vite, and the latest web technologies for optimal performance.',
      icon: '🏗️'
    },
    {
      title: 'Secure Authentication',
      description: 'Protected routes and secure user authentication using Context API.',
      icon: '🔐'
    },
    {
      title: 'Responsive Design',
      description: 'Fully responsive layout that works seamlessly on all devices.',
      icon: '📱'
    },
    {
      title: 'Beautiful UI',
      description: 'Stunning user interface designed with Tailwind CSS and modern aesthetics.',
      icon: '🎨'
    }
  ]

  return (
    <div className="min-h-[calc(100vh-4rem)] px-4 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            About <span className="bg-gradient-to-r from-purple-400 to-pink-500 text-transparent bg-clip-text">AppName</span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            We're on a mission to revolutionize the way people interact with web applications. 
            Our platform combines cutting-edge technology with beautiful design to deliver 
            exceptional user experiences.
          </p>
        </div>

        {/* Mission Section */}
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 md:p-12 border border-purple-500/20 mb-12">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">Our Mission</h2>
              <p className="text-gray-300 leading-relaxed mb-4">
                We believe in creating technology that empowers people and businesses to achieve 
                their goals. Our platform is designed to be intuitive, powerful, and accessible 
                to everyone.
              </p>
              <p className="text-gray-300 leading-relaxed">
                With a focus on innovation and user experience, we're constantly pushing the 
                boundaries of what's possible in web development.
              </p>
            </div>
            <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl p-8 border border-purple-500/30">
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-purple-500/30 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">🚀</span>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold">Innovation First</h3>
                    <p className="text-gray-400 text-sm">Always pushing boundaries</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-pink-500/30 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">💡</span>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold">User Focused</h3>
                    <p className="text-gray-400 text-sm">Your needs drive our development</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-purple-500/30 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">⚡</span>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold">Performance</h3>
                    <p className="text-gray-400 text-sm">Lightning-fast experiences</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-white text-center mb-12">Why Choose Us</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-200 hover:scale-105"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Team Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white text-center mb-12">Meet Our Team</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {team.map((member, index) => (
              <div
                key={index}
                className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20 text-center hover:border-purple-500/40 transition-all duration-200 hover:scale-105"
              >
                <div className="text-6xl mb-4">{member.avatar}</div>
                <h3 className="text-xl font-bold text-white mb-1">{member.name}</h3>
                <p className="text-purple-400 text-sm">{member.role}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Stats Section */}
        <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl p-8 md:p-12 border border-purple-500/20">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 text-transparent bg-clip-text mb-2">
                10K+
              </div>
              <p className="text-gray-300">Active Users</p>
            </div>
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 text-transparent bg-clip-text mb-2">
                50+
              </div>
              <p className="text-gray-300">Countries</p>
            </div>
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 text-transparent bg-clip-text mb-2">
                99.9%
              </div>
              <p className="text-gray-300">Uptime</p>
            </div>
            <div>
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 text-transparent bg-clip-text mb-2">
                24/7
              </div>
              <p className="text-gray-300">Support</p>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center mt-16">
          <h2 className="text-3xl font-bold text-white mb-4">Ready to Get Started?</h2>
          <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
            Join thousands of users who are already experiencing the future of web applications.
          </p>
          <button className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-purple-500/50 hover:scale-105">
            Contact Us Today
          </button>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard