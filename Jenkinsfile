pipeline {
    agent any
    environment {
        APP_NAME = "FeedbackManagementSystem"
    }
    stages {
        stage('Clone Repository') {
            steps {
                checkout scm
                
                withCredentials([
                    string(credentialsId: 'fms', variable: 'GROQ_API_KEY')
                ]) {
                    sh '''
                        touch core/.env
                        echo "GROQ_API_KEY=${GROQ_API_KEY}" > core/.env
                    '''
                } 
            }
        }
        
        stage('Docker-compose-down') {
            steps {
                sh 'cd core && docker-compose down --remove-orphans'
                echo "Taking down all containers and removing orphans"
            }
        }   
        
        stage('Docker-compose-up') {
            steps {
                echo "Creating docker containers"
                sh '''
                    cd core
                    docker-compose up -d --build
                '''
            }
        }
    }
    
    post {
        success {
            echo "Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}