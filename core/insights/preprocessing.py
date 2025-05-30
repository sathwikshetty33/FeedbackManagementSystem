from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from home.models import Event
from django.core.mail import send_mail
from django.conf import settings
# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler 
from langchain.callbacks.manager import CallbackManager
from django.utils import timezone
import re
from functools import lru_cache
import hashlib
from io import StringIO
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler(), logging.FileHandler('feedback_insights_langchain.log')])
logger = logging.getLogger(__name__)

