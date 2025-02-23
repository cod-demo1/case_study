import streamlit as st


st.markdown(
    f"""
    #### :exclamation: Case Study Part I.
    ### Automating Software Quality Control with AI.
    #####
    """
)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #### **Problem Summary**
        - Suppliers make various software changes, not always the requested ones.
        - Identifying all code changes and verifying them against requests is challenging.
        - Unexpected changes cause difficult-to-resolve problems.
        """
    )
with c2:
    st.markdown(
        f"""
        #### **Project Phases**
        
        **Phase 1**: Develop or use existing software to compare source codes and identify all changes.
        
        **Phase 2**: Automatically load and understand change request descriptions, compare them with the identified
         code changes, and evaluate whether there were any unexpected changes.
        """
    )
st.markdown(
    f"""
    ####
    #### :exclamation: 1. Team and Scope
    """
)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #####
        ##### Team Roles:
        
        - ***Project Manager*** – Oversees project scope, timeline, and deliverables.
        - ***AI/ML Engineer*** – Integrates LLM models for change request analysis.
        - ***Software Engineer*** - Works on AST parsing and change detection.
        - ***Data/DevOps Engineer*** – Sets up infrastructure, CI/CD, and automation.
        - ***QA Engineer*** – Validates system accuracy and reliability.
        
        ###
        ##### Challenges
    
        - Handling multiple / complex programming languages (esp. AST parsing).
        - Data inconsistency (e.g., incorrect or incomplete diffs).
        - Ensuring LLM models correctly interpret vague change requests (RAG / model fine-tuning).
        - LLMs are effective but can hallucinate, making the process susceptible to errors.
        - Scalability to large codebases (efficient diffing and processing, models).
        
        """
    )
with c2:
    st.markdown(
        f"""
        #####
        ##### Estimated Timeline and Key Milestones
        
        1. Planning	Gather requirements, define scope [2 weeks]
        2. **Phase 1**:	Develop & test source code comparator [4-6 weeks]
            - Implement Version Control Parsing & AST Analysis [2-3 weeks]
            - Implement LLM based change summarization [2-3 weeks]
            - Testing & Validation of Comparison Tool [2 weeks]
        3. **Phase 2**:	Implement LLM-based request verification [5-7 weeks]
            - Develop LLM based Request Parsing [2-4 weeks]
            - Develop Matching Algorithms for Requests vs Code Changes [2-4 weeks]
            - Testing, Validation & PoC Evaluation [2 weeks]
        4. (optional) **Phase 3**: Automated Code Review & Suggestions (LLM) [4-6 weeks]
            - Detect potential conflicts in modified code.
            - Provide structured comments on GitHub PR.
            - Suggest improvements based on best practices.
        
        """
    )

st.markdown(
    f"""
    ####
    #### :exclamation: 2. Information Gathering
    
    #####
    ##### Key Questions for the Client
    
    - Codebase Details: What languages are used? Is there a version control system (e.g., Git)?
        - -> Parsing approach
    - Change Request Format: Are requests structured (e.g., Jira tickets, JSON) or unstructured (e.g., emails, PDFs)?
        - -> NLP / LLM model choice & design
    - Scope of Changes: function-level, file-level, or system-wide modifications?
        - -> Scalability
    - Expected integration points (CI/CD pipelines, repositories)
        - -> Deployment
    - Validation Criteria: How does the client currently validate (expected vs. unexpected) changes?
    - Security Constraints: Any restrictions on AI model usage and data handling.
        
    
    ####
    #### :exclamation: 3. Data and Format
    """
)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #####
        ##### Source Code
        - Format: plain text files of specific programming language
        - Extent: Git diff outputs
        - Preprocessing: AST parsing
        
        
        ##### 
        ##### Pre-processing Steps

        - Source Code:
            - Convert source files into AST representation for better structural comparison
            - Parse Git diff logs to extract changed code snippets
        - Change Requests:
            - Convert unstructured text to structured representations.
        - Embeddings:
            - Generate vector representations for semantic matching.
            
        """
    )
with c2:
    st.markdown(
        f"""
        #####
        ##### Change Request Descriptions:
        - Format:
            - Structured (JSON, XML)
            - Unstructured (plain text, PDFs, emails).
        - Preprocessing: 
            - OCR for scanned documents
            - Text normalization
            - Keyword extraction
        """
    )

st.markdown(
    f"""
    ####
    #### :exclamation: 4. Comparison Methods
    """
)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #####
        ##### Phase 1: Source Code Comparison
        
        1. Git diff outputs
            - Textual comparison, line-by-line changes
        2. Abstract Syntax Tree (AST) Analysis
            - Structural function/class changes
        3. LLM-assisted Code Understanding
            - Summarize changes
            - Classify modifications
            - Explain impact
        
        """
    )
with c2:
    st.markdown(
        f"""
        #####
        ##### Phase 2: Comparing Code Changes to Requests
        
        1. Text Embeddings (OpenAI ADA, Sentence Transformers)
            - Vector representations of natural language
            - Captures semantic meaning
            - Requires high-quality embeddings
        2. BM25 / Keyword extraction
            - Finds exact match keywords between change requests & code
        3. LLM-based Comparison
            - Use an LLM to analyze if changes align with request
            - Deep understanding of context
            - Computationally expensive
        - -> Hybrid Approach
        """
    )

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #### -> Approach design
        ######
        
        ##### Existing LLMs already understand programming languages and natural language.
        ##### Solution: A hybrid combination of RAG and structured matching to enhance accuracy.
        ######
        
        ##### 📍 Step 1: Extract Actual Code Changes
        1. Git Diff & AST Analysis
            - Identify textual and structural modifications within source code.
        
        ##### 
        ##### 📍 Step 2: Extract Key Information from the Change Request (LLM)
        1. Convert vague natural language into a structured, actionable breakdown.
            - Extract intent: Determine if the request is a refactor, bug fix, or feature addition.
            - Identify expected impact (e.g., "Should modify class A but not class B").
        
        ##### 
        ##### 📍 Step 3: Retrieve Past Relevant PRs / changes for Context (RAG)
        1. Extract relevant historical change requests (from PRs, issue trackers, commit messages).
        2. Retrieve similar past changes from a vector database to enhance verification.

        ##### 
        ##### 📍 Step 4: Retrieve Relevant Codebase Parts (Vector DB)
        1. Identify related code snippets that may be impacted by new changes.
        2. Extract relevant function definitions, class structures, and key modules.
        
        ##### 
        ##### 📍 Step 5: LLM-Powered Code Review with Full Context
        -> **Provide retrieved examples as context to the LLM.**
        
        - Prevents breaking dependencies, existing code reviewed before suggesting changes.
        - Avoids redundant code, avoids duplicating existing logic.
        - Improves accuracy, embedding search reduces LLM hallucinations.
                
        """
    )
with c2:
    st.markdown(
        """
        #
        #
        #
        #
        
        ```
        Analyze the following change request and extract structured details:
        - What functionality is being changed?
        - What files or components should be modified?
        - What expected behavior should be verified?
    
        Change Request: {change_request}
        ```
        #
        #
        
        ```
        You are reviewing a code change in a GitHub PR.

        **Change Request:**  
        {change_request}
    
        **Relevant Past Changes:**  
        {past_changes}
    
        **Existing Codebase Context (Relevant Functions & Modules):**  
        {relevant_code}
    
        **Actual Code Diff:**  
        {code_diff}
    
        **Review Criteria:**  
        - Does the actual code change fully address the change request?
        - Does it properly integrate with existing code?
        - Are there any issues, inefficiencies, or inconsistencies?
        - Does it follow best practices based on similar past changes?
    
        Provide a structured review with any concerns or recommendations.
        ```
        
        
        
        
        """
    )

st.markdown(
    """
    ####
    #### :exclamation: 5. Technology Stack
    """
)
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        #####
        ##### Languages & Tools:
        
        - Code Diff Analysis: Python (difflib, AST), Git
        - AI/ML Frameworks: GPT-style model APIs, open-source LLMs, Hugging Face transformers
        - Text Embeddings & Retrieval: vector DB (Pinecone, LanceDB)
        - Backend & API: FastAPI, Flask
        - MLOps & CI/CD: Docker, Kubernetes, MLflow, DVC
        - Infrastructure: AWS/GCP for scalability
        
        #####
        ##### Architecture Diagram:
        
        1. Developer submits PR
        2. **Phase 1**: Code analysis
            - Identify syntactic changes (AST analysis, Git Diff parsing).
            - Identify semantic changes (LLM-powered change summarization).
        3. **Phase 2**: LLM comparison to Descriptions
            - Match requested vs. actual changes (Text embeddings, LLMs).
            - Retrieve past relevant PRs (Vector DB search, RAG).
        4. Generate review
            - Integrate github Bot for auto PR comments / approval rules.
            - Propose AI-generated fixes.
            - Integrate into CI/CD DevOps pipelines.
        """
    )
with c2:
    st.markdown(
        f"""
        #####
        ##### Infrastructure Requirements:

        ###### PoC Phase
        1. Compute: CPU-based VMs
        2. LLM API access
        3. vector DB (cloud-hosted or self-managed)
        4. FastAPI / Flask backend
        5. Deployment: containerization (docker, k8s)

        ###### Production Phase
        1. Compute: scaling GPU-based VMs / k8s cluster (LLM scaling)
        2. CI / CD integration (GitHub Actions) - PR pipelines
        3. Extend metadata stores (PostgreSQL DB, vector DB)
        4. MLOps pipeline (MLFlow) - model versioning, embeddings, deployment

        """
    )

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        f"""
        ####
        #### :exclamation: 6. MLOps Considerations
        
        #####
        ##### For Maintainability & Scalability:
        
        - Model & Data versioning: MLflow / DVC.
        - Store trained models, embeddings, and metadata in a centralized registry for easy rollback.
        - Continuous Integration: Automated testing and validation of models.
            - Validate LLM outputs against historical PRs & standard datasets.
            - Run unit tests for AST parsing, vector retrieval, and LLM-based validation.
        - Logging & Monitoring: Real-time tracking of model performance.
            - Track accuracy, precision, recall for LLM-driven validation.
            - Prometheus/Grafana for real-time performance tracking.
        - Scaling & Infrastructure
            - Optimize LLM calls via caching mechanisms for previous queries.
            - Use lightweight embedding models where possible to reduce inference costs.
        - Compute Resource Optimization.
            - Leverage serverless functions (AWS Lambda, GCP Functions) for on-demand processing.
        - Security.
            - Access Control: Restrict model API access using OAuth, API keys, or IAM roles.
        """
    )
with c2:
    st.markdown(
        f"""
        ####
        #### :exclamation: 7. Knowledge Transfer
        
        #####
        ##### Internal Team Collaboration & Communication:
        
        - Technical Documentation: Architecture, API specs, model details.
        - Code Review Practices: Well-documented GitHub repository & knowledge transfer through collaborative code reviews.
        - Training Sessions: Hands-on workshops for model usage and debugging.
        
        #####
        ##### Client Engagement & Training:
        
        - Document key client requirements and preferences to avoid scope misalignment.
        - Executive Summary: High-level report for non-technical stakeholders.
        - Detailed Technical Documents: Detailed documentation for in-house teams.
        - Training Sessions: Hands-on workshops and post-deployment training for clients.
        
        #####
        ##### Final Deliverables:
        
        - Working PoC with source code and AI models.
        - Reports detailing findings, accuracy metrics, and recommendations.
        - Plan for production deployment and long-term support.
        - Feedback Loops: Regular feedback sessions from both clients and end-users to refine the system.
        
        """
    )
