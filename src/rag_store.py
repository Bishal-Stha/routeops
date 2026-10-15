"""Domain Knowledge RAG Vector Store for RouteOps.

Provides vector retrieval and snippet generation over 25-30 comprehensive domain-specific
documents covering eSewa/Khalti troubleshooting, NTC/Ncell data pack activation,
local banking FAQs (fixed deposits, KYC, loans), and technical API rate limit rules
using NumPy cosine similarity and the zero-extra-model n-gram projection vectorizer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.router_service import compute_ngram_projection_embedding

logger = logging.getLogger("routeops.rag_store")


class RAGDocument(BaseModel):
    """Structured representation of a RAG knowledge base document."""

    doc_id: str
    title: str
    category: str  # esewa_khalti, ntc_ncell, banking_kyc, api_limits
    content: str
    keywords: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Curated Domain Knowledge Base (28 Documents)
# -----------------------------------------------------------------------------

CURATED_DOMAIN_DOCS: List[Dict[str, Any]] = [
    # --- Category 1: eSewa & Khalti Troubleshooting (Docs 1-7) ---
    {
        "doc_id": "esewa_001",
        "title": "eSewa Transaction Pending or Failed Refund Policy",
        "category": "esewa_khalti",
        "content": (
            "If an eSewa transaction shows 'Pending' or 'Processing' but money was debited "
            "from your linked bank account, do not repeat the transaction immediately. "
            "Automated system reconciliation runs every 15 minutes. If funds are not credited "
            "or refunded within 24 hours, contact eSewa toll-free support at 1660-01-55555 "
            "or raise a support ticket with your Transaction ID (e.g., ESW984123)."
        ),
        "keywords": ["esewa", "pending", "refund", "transaction", "debited", "failed"],
    },
    {
        "doc_id": "esewa_002",
        "title": "Khalti MPIN Reset and Wallet Locking Security",
        "category": "esewa_khalti",
        "content": (
            "Entering an incorrect Khalti MPIN 3 consecutive times temporarily locks your wallet "
            "for 30 minutes to prevent unauthorized access. To reset your MPIN instantly, tap "
            "'Forgot MPIN' on the login screen, enter your registered 10-digit mobile number, "
            "and verify with the 6-digit OTP sent via SMS."
        ),
        "keywords": ["khalti", "mpin", "locked", "reset", "otp", "security"],
    },
    {
        "doc_id": "esewa_003",
        "title": "eSewa KYC Verification and Transaction Limits",
        "category": "esewa_khalti",
        "content": (
            "Unverified eSewa accounts have a monthly transaction limit of NRs. 10,000. "
            "To upgrade to a Full KYC Verified account, submit a government-issued photo ID "
            "(Citizenship, Passport, or National ID/Nagarikta), a live selfie, and fill in "
            "your permanent address. Full KYC accounts enjoy a daily limit of NRs. 1,00,000."
        ),
        "keywords": ["esewa", "kyc", "limit", "citizenship", "verified", "upgrade"],
    },
    {
        "doc_id": "esewa_004",
        "title": "Khalti Load Fund via ConnectIPS and Mobile Banking",
        "category": "esewa_khalti",
        "content": (
            "You can load funds into your Khalti wallet directly via ConnectIPS, Visa/Mastercard, "
            "or partner bank mobile banking apps (Nabil, NIC Asia, Global IME). ConnectIPS loads "
            "are free of charge for amounts above NRs. 500, while direct card loads may incur "
            "a 0.75% gateway processing fee."
        ),
        "keywords": ["khalti", "load", "connectips", "bank", "fee", "transfer"],
    },
    {
        "doc_id": "esewa_005",
        "title": "eSewa Send Money to Bank Account Transfer Timings",
        "category": "esewa_khalti",
        "content": (
            "Transfers from eSewa to commercial bank accounts via ACH/ConnectIPS are processed "
            "instantly 24/7 for participating members of the National Payment Switch. "
            "Interbank transfers exceeding NRs. 1,00,000 may require up to 2 business hours "
            "for clearance during Nepal Clearing House (NCHL) working days."
        ),
        "keywords": ["esewa", "bank", "transfer", "nchl", "connectips", "instant"],
    },
    {
        "doc_id": "esewa_006",
        "title": "Khalti Cashback and Reward Points Redemption",
        "category": "esewa_khalti",
        "content": (
            "Khalti users earn 0.5% to 2% cashback on utility payments (electricity, water, "
            "internet bills) and Khalti Points (KP) on movie tickets. Reward points can be "
            "redeemed at checkout when accumulated points exceed 100 KP (100 KP = NRs. 1)."
        ),
        "keywords": ["khalti", "cashback", "reward", "points", "utility", "redemption"],
    },
    {
        "doc_id": "esewa_007",
        "title": "Resolving eSewa Merchant API Integration Authentication Errors",
        "category": "esewa_khalti",
        "content": (
            "When integrating eSewa ePay payment gateway, ensure your Merchant Secret Key "
            "is transmitted using HMAC-SHA256 signature hashing. Invalid signature errors "
            "typically stem from incorrect field ordering in the signature payload string "
            "or mismatched total_amount and transaction_uuid values."
        ),
        "keywords": ["esewa", "merchant", "api", "hmac", "signature", "integration"],
    },
    # --- Category 2: NTC & Ncell Telecom Data Pack Activation (Docs 8-14) ---
    {
        "doc_id": "ntc_001",
        "title": "Nepal Telecom (NTC) All-Time Data Pack Activation Codes",
        "category": "ntc_ncell",
        "content": (
            "To activate NTC data packs, dial *1415# from your GSM/4G SIM or use the "
            "Nepal Telecom mobile app. Popular packs include the 1GB 24-hour pack for NRs. 28 "
            "(dial *1415*1*1#) and the 7-day 3GB unlimited night pack for NRs. 98."
        ),
        "keywords": ["ntc", "nepal telecom", "data", "pack", "dial", "activation"],
    },
    {
        "doc_id": "ntc_002",
        "title": "Ncell Freedom and Daily Social Data Packs",
        "category": "ntc_ncell",
        "content": (
            "Ncell subscribers can subscribe to social media data packs (TikTok, Facebook, Instagram) "
            "by dialing *17123# or via the Ncell App. The 7-day social pack providing 5GB allocation "
            "is priced at NRs. 135 inclusive of all applicable telecom taxes."
        ),
        "keywords": ["ncell", "data", "social", "tiktok", "pack", "dial"],
    },
    {
        "doc_id": "ntc_003",
        "title": "Checking Balance and Data Volume Quota on NTC and Ncell",
        "category": "ntc_ncell",
        "content": (
            "To check remaining main balance on NTC, dial *400#. To check active data pack volume "
            "and expiry, dial *1415*55#. For Ncell, dial *101# for main balance and *901# "
            "or *17123# option 4 to check remaining data volume."
        ),
        "keywords": ["balance", "data", "quota", "ntc", "ncell", "check"],
    },
    {
        "doc_id": "ntc_004",
        "title": "NTC VoLTE Activation and Compatible Handsets",
        "category": "ntc_ncell",
        "content": (
            "Nepal Telecom Voice over LTE (VoLTE) offers high-definition crystal clear voice calls "
            "over the 4G network. To activate VoLTE, dial *444# on an NTC postpaid or prepaid SIM. "
            "Ensure your smartphone has VoLTE enabled and your APN settings are configured to 'ntnet'."
        ),
        "keywords": ["ntc", "volte", "4g", "voice", "activation", "apn"],
    },
    {
        "doc_id": "ntc_005",
        "title": "Ncell International Roaming Activation in India and China",
        "category": "ntc_ncell",
        "content": (
            "Ncell prepaid and postpaid users can activate international roaming before traveling "
            "abroad by visiting an Ncell Center or dialing *171*1#. For India roaming packs, "
            "special 7-day incoming and outgoing bundled voice/data rates apply starting at NRs. 999."
        ),
        "keywords": ["ncell", "roaming", "india", "international", "travel", "pack"],
    },
    {
        "doc_id": "ntc_006",
        "title": "NTC FTTH Fiber Internet and IPTV Combo Packages",
        "category": "ntc_ncell",
        "content": (
            "Nepal Telecom Fiber to the Home (FTTH) offers high-speed broadband internet combined "
            "with IPTV services. Standard speeds range from 50 Mbps to 250 Mbps with annual "
            "subscription packages starting at NRs. 13,000 excluding 13% Telecommunications Service Charge (TSC)."
        ),
        "keywords": ["ntc", "ftth", "fiber", "internet", "broadband", "iptv"],
    },
    {
        "doc_id": "ntc_007",
        "title": "Troubleshooting NTC/Ncell Slow 4G Internet Speed and APN",
        "category": "ntc_ncell",
        "content": (
            "If your mobile data connection is slow, verify that your Access Point Name (APN) "
            "is set to 'web' for Ncell or 'ntnet' for NTC. Toggling airplane mode on and off "
            "or switching network preference from 4G/3G/2G auto to LTE-only mode often resolves "
            "handshake latency issues."
        ),
        "keywords": ["slow", "internet", "apn", "4g", "network", "troubleshoot"],
    },
    # --- Category 3: Local Banking, Fixed Deposits, KYC & Loans (Docs 15-22) ---
    {
        "doc_id": "bank_001",
        "title": "Commercial Bank Fixed Deposit (FD) Interest Rates in Nepal",
        "category": "banking_kyc",
        "content": (
            "As per Nepal Rastra Bank (NRB) directives, commercial banks offer competitive "
            "interest rates on individual Fixed Deposits ranging from 6.5% to 8.5% per annum, "
            "compounded quarterly. Senior citizens typically receive an additional 0.5% premium "
            "on fixed term deposits of 1 year or more."
        ),
        "keywords": ["bank", "fixed deposit", "fd", "interest", "rate", "nrb"],
    },
    {
        "doc_id": "bank_002",
        "title": "Required Documents for Opening a Savings Bank Account in Nepal",
        "category": "banking_kyc",
        "content": (
            "To open a bank account (savings or current) in Nepal, applicants must provide: "
            "(1) Original and photocopies of Citizenship Certificate (Nagarikta), Passport, or Voter ID, "
            "(2) Two recent passport-size photographs, (3) PAN card copy (mandatory for interest tax exemption "
            "and digital banking above threshold), and (4) Duly filled KYC application form with nominee details."
        ),
        "keywords": ["bank", "account", "savings", "citizenship", "kyc", "documents"],
    },
    {
        "doc_id": "bank_003",
        "title": "Home Loan and SME Loan Eligibility Criteria",
        "category": "banking_kyc",
        "content": (
            "Commercial banks in Nepal offer home loans up to 70% of property valuation (Fair Market Value) "
            "with repayment tenures up to 25 years. Applicants must demonstrate steady verified income "
            "(salary certificate, audited business financials, or rental agreements) with a Debt-to-Income "
            "(DTI) ratio not exceeding 50% of net monthly earnings."
        ),
        "keywords": ["home loan", "sme", "mortgage", "eligibility", "income", "bank"],
    },
    {
        "doc_id": "bank_004",
        "title": "Demat Account and MeroShare Registration Procedure",
        "category": "banking_kyc",
        "content": (
            "To trade shares on NEPSE (Nepal Stock Exchange), investors must maintain a Demat account "
            "and C-ASBA registered bank account. MeroShare credentials for IPO applications are issued "
            "by your Depository Participant (DP / Capital Market branch) upon payment of an annual "
            "renewal fee of NRs. 50."
        ),
        "keywords": ["demat", "meroshare", "nepse", "ipo", "share", "c-asba"],
    },
    {
        "doc_id": "bank_005",
        "title": "Withholding Tax on Bank Interest and Dividend Income",
        "category": "banking_kyc",
        "content": (
            "In accordance with the Inland Revenue Department (IRD) tax laws of Nepal, "
            "a flat 5% withholding tax is deducted at source on interest earned from bank savings "
            "and fixed deposit accounts. Dividend income distributed by companies and mutual funds "
            "is also subject to a 5% final withholding tax."
        ),
        "keywords": ["tax", "interest", "dividend", "withholding", "ird", "bank"],
    },
    {
        "doc_id": "bank_006",
        "title": "Credit Card Eligibility and Dollar Card Facility in Nepal",
        "category": "banking_kyc",
        "content": (
            "Nepali commercial banks issue pre-paid Dollar Cards (up to $500 equivalent per fiscal year) "
            "for international e-commerce transactions upon submitting a valid PAN card and active bank account. "
            "Credit cards require a minimum monthly salary threshold of NRs. 40,000 or a blocked "
            "Fixed Deposit collateral equivalent to 120% of the requested credit limit."
        ),
        "keywords": [
            "credit card",
            "dollar card",
            "pan",
            "international",
            "bank",
            "limit",
        ],
    },
    {
        "doc_id": "bank_007",
        "title": "Banking Mobile App OTP Timeout and Device Binding Issues",
        "category": "banking_kyc",
        "content": (
            "Most Nepali mobile banking apps (e.g., NIC Asia Mobile Banking, Kumari Smart) enforce "
            "strict device binding for security. If you change your smartphone or reinstall the app, "
            "you must re-verify your identity via debit card number, PIN, and SMS OTP, or visit "
            "your home branch to unbind the previous device MAC address."
        ),
        "keywords": [
            "mobile banking",
            "otp",
            "device binding",
            "app",
            "security",
            "bank",
        ],
    },
    {
        "doc_id": "bank_008",
        "title": "Locker Facility Services and Annual Rental Charges",
        "category": "banking_kyc",
        "content": (
            "Safe deposit locker facilities are available at select commercial bank branches "
            "for storing gold ornaments and legal deeds. Annual rental charges vary by locker size "
            "(Small: NRs. 3,000 - 5,000; Medium: NRs. 6,000 - 10,000; Large: NRs. 12,000 - 20,000) "
            "and require maintaining a minimum balance savings account."
        ),
        "keywords": ["locker", "safe", "deposit", "gold", "rental", "bank"],
    },
    # --- Category 4: Technical API Rate Limits & Gateway Rules (Docs 23-28) ---
    {
        "doc_id": "api_001",
        "title": "RouteOps Semantic AI Gateway REST API Rate Limit Policy",
        "category": "api_limits",
        "content": (
            "The RouteOps /v1/route REST API enforces a standard rate limit of 120 requests per minute "
            "per API key for standard tier clients, and 600 requests per minute for enterprise tier clients. "
            "Exceeding this quota returns an HTTP 429 Too Many Requests response with a 'Retry-After' "
            "header specifying seconds until quota window reset."
        ),
        "keywords": ["api", "rate limit", "throttle", "429", "requests", "gateway"],
    },
    {
        "doc_id": "api_002",
        "title": "ONNX Runtime Inference Timeout and Thread Allocation Guidelines",
        "category": "api_limits",
        "content": (
            "To respect the hardware envelope of the Intel Core i5-12450H (4 P-cores), ONNX Runtime "
            "inference sessions must be configured with intra_op_num_threads=4 and inter_op_num_threads=1. "
            "Maximum inference execution time for multilingual classification must not exceed P95 < 15ms. "
            "Requests timing out past 3000ms trigger an automatic fallback route."
        ),
        "keywords": ["onnx", "runtime", "threads", "timeout", "latency", "p95"],
    },
    {
        "doc_id": "api_003",
        "title": "Semantic Cache Eviction and TTL Configuration Rules",
        "category": "api_limits",
        "content": (
            "The RouteOps in-memory and SQLite semantic cache utilizes cosine similarity matching "
            "with a default threshold of 0.92. Cache entries have a Time-To-Live (TTL) of 24 hours "
            "after which stale embeddings are pruned. When the cache table exceeds 10,000 entries, "
            "LRU (Least Recently Used) eviction based on hit_count and timestamp is automatically executed."
        ),
        "keywords": [
            "semantic cache",
            "ttl",
            "cosine",
            "similarity",
            "eviction",
            "sqlite",
        ],
    },
    {
        "doc_id": "api_004",
        "title": "PII Redaction Regex Patterns and Privacy Compliance",
        "category": "api_limits",
        "content": (
            "RouteOps Gateway enforces strict zero-retention PII sanitization. Incoming queries "
            "are scanned for Nepali phone numbers (+977 98XXXXXXXX), OTP codes (4-6 digits), "
            "API keys (sk-, ghp_, Bearer), and 16-digit credit card sequences prior to logging "
            "or upstream model dispatch. Redacted entities are replaced with standardized tags "
            "like [REDACTED_PHONE] or [REDACTED_OTP]."
        ),
        "keywords": ["pii", "redaction", "privacy", "regex", "sanitize", "compliance"],
    },
    {
        "doc_id": "api_005",
        "title": "Population Stability Index (PSI) Drift Monitoring Thresholds",
        "category": "api_limits",
        "content": (
            "The drift monitor evaluates incoming query embedding shifts and confidence distributions "
            "against baseline reference sets using Population Stability Index (PSI) and Jensen-Shannon Divergence. "
            "A PSI score below 0.10 indicates stable operations; 0.10 <= PSI < 0.25 triggers a warning alert; "
            "and PSI >= 0.25 signals significant concept drift requiring automated model fine-tuning."
        ),
        "keywords": [
            "psi",
            "drift",
            "monitoring",
            "stability",
            "jensen-shannon",
            "mlops",
        ],
    },
    {
        "doc_id": "api_006",
        "title": "Confidence Abstention Cascade and Unit Economics Optimization",
        "category": "api_limits",
        "content": (
            "When model classification confidence falls below 0.70 (or Shannon entropy exceeds 1.5 bits), "
            "RouteOps triggers a confidence abstention cascade, routing the request to the frontier reasoning "
            "tier to ensure safety and accuracy. Unit economic savings are quantified by comparing actual "
            "route costs against the naive baseline cost of running every query on the frontier LLM."
        ),
        "keywords": [
            "confidence",
            "abstention",
            "cascade",
            "entropy",
            "economics",
            "cost",
        ],
    },
]


class RAGStore:
    """High-performance local RAG knowledge base utilizing NumPy and cosine similarity."""

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize RAGStore with documents and precompute their L2-normalized embeddings.

        Args:
            documents: Optional custom list of document dicts. Defaults to CURATED_DOMAIN_DOCS.
        """
        self.raw_docs = documents if documents is not None else CURATED_DOMAIN_DOCS
        self.documents: List[RAGDocument] = []
        self.embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)

        self._build_index()

    def _build_index(self) -> None:
        """Parse raw documents into RAGDocument models and precompute normalized embedding matrix."""
        doc_models: List[RAGDocument] = []
        vecs: List[np.ndarray] = []

        for d in self.raw_docs:
            doc_model = RAGDocument(
                doc_id=d["doc_id"],
                title=d["title"],
                category=d["category"],
                content=d["content"],
                keywords=d.get("keywords", []),
            )
            doc_models.append(doc_model)

            # Compute embedding on title + content for rich semantic coverage
            combined_text = f"{doc_model.title} {doc_model.content}"
            emb = compute_ngram_projection_embedding(combined_text, dim=128)
            vecs.append(emb)

        self.documents = doc_models
        if vecs:
            self.embeddings = np.vstack(vecs).astype(np.float32)
        else:
            self.embeddings = np.empty((0, 128), dtype=np.float32)

        logger.info(
            "Initialized RAGStore with %d documents and embedding matrix shape %s",
            len(self.documents),
            self.embeddings.shape,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        min_similarity: float = 0.15,
    ) -> List[Tuple[RAGDocument, float]]:
        """Retrieve top-k most relevant documents using cosine similarity over n-gram projections.

        Args:
            query: User query string.
            top_k: Maximum number of documents to return (default: 3).
            category: Optional category filter ('esewa_khalti', 'ntc_ncell', 'banking_kyc', 'api_limits').
            min_similarity: Minimum cosine similarity threshold (default: 0.15).

        Returns:
            List of tuples containing (RAGDocument, cosine_similarity_score).
        """
        if not self.documents or self.embeddings.size == 0:
            return []

        q_vec = compute_ngram_projection_embedding(query, dim=128)
        norm_q = float(np.linalg.norm(q_vec))
        if norm_q > 1e-8:
            q_vec /= norm_q

        # Compute cosine similarities via matrix multiplication (both are L2 normalized)
        similarities = np.dot(self.embeddings, q_vec)

        scored_docs: List[Tuple[RAGDocument, float]] = []
        for idx, sim in enumerate(similarities):
            doc = self.documents[idx]
            if category and doc.category != category:
                continue
            sim_score = float(sim)
            if sim_score >= min_similarity:
                scored_docs.append((doc, sim_score))

        # Sort by similarity descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def add_document(
        self, doc_id: str, title: str, category: str, content: str, keywords: List[str]
    ) -> None:
        """Add a new document to the RAG store and update the embedding index."""
        new_dict = {
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "content": content,
            "keywords": keywords,
        }
        self.raw_docs.append(new_dict)
        self._build_index()
