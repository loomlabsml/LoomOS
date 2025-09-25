"""
LoomOS Marketplace - AI Compute Advisory & Recommendation System

The Marketplace is LoomOS's intelligent advisory layer that provides:
- Real-time compute pricing discovery across providers
- Guided resource selection and configuration
- Step-by-step provisioning instructions
- Cost estimation and budget planning
- Provider comparison and recommendations
- Community-driven reviews and ratings
- Training job templates and best practices
- Educational resources for optimal setup

Architecture:
- Provider API integrations for live pricing
- Intelligent recommendation algorithms
- User-guided configuration workflows
- Educational content and tutorials
- Community knowledge sharing platform
- Cost tracking and budget alerts
- Security best practices guidance
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from abc import ABC, abstractmethod
from prometheus_client import Counter, Histogram, Gauge

# Mock payment and blockchain libraries
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    # Mock Stripe
    class stripe:
        class PaymentIntent:
            @staticmethod
            def create(**kwargs): return {"id": "pi_test", "status": "succeeded"}

logger = logging.getLogger(__name__)

# Metrics
MARKETPLACE_TRANSACTIONS = Counter('loomos_marketplace_transactions_total', 'Total marketplace transactions', ['type', 'status'])
REVENUE_GENERATED = Counter('loomos_marketplace_revenue_usd', 'Total revenue generated in USD')
ACTIVE_LISTINGS = Gauge('loomos_marketplace_active_listings', 'Number of active marketplace listings')
COMPUTE_UTILIZATION = Gauge('loomos_marketplace_compute_utilization', 'Compute resource utilization')
TRANSACTION_LATENCY = Histogram('loomos_marketplace_transaction_seconds', 'Transaction processing time')

class CurrencyType(Enum):
    """Supported currency types"""
    USD = "usd"
    LOOM_CREDITS = "loom_credits"
    COMPUTE_UNITS = "compute_units"
    BTC = "btc"
    ETH = "eth"

class ListingType(Enum):
    """Types of marketplace listings"""
    COMPUTE_RECOMMENDATION = "compute_recommendation"
    CONFIGURATION_GUIDE = "configuration_guide"
    TRAINING_TEMPLATE = "training_template"
    COST_CALCULATOR = "cost_calculator"
    PROVIDER_COMPARISON = "provider_comparison"
    TUTORIAL = "tutorial"
    COMMUNITY_GUIDE = "community_guide"

class RecommendationStatus(Enum):
    """Status of marketplace recommendations"""
    DRAFT = "draft"
    ACTIVE = "active"
    VERIFIED = "verified"
    OUTDATED = "outdated"
    COMMUNITY_REVIEWED = "community_reviewed"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"

class BillingCycle(Enum):
    """Billing cycle options"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"
    PAY_PER_USE = "pay_per_use"

@dataclass
class Price:
    """Price representation with multiple currencies"""
    amount: Decimal
    currency: CurrencyType
    billing_cycle: BillingCycle = BillingCycle.PAY_PER_USE
    
    def to_usd(self, exchange_rates: Dict[CurrencyType, Decimal]) -> Decimal:
        """Convert to USD using exchange rates"""
        if self.currency == CurrencyType.USD:
            return self.amount
        
        rate = exchange_rates.get(self.currency, Decimal('1.0'))
        return self.amount * rate
    
    def __str__(self):
        return f"{self.amount} {self.currency.value}"

@dataclass
class Account:
    """User account with balances and billing info"""
    user_id: str
    email: str
    
    # Balances
    balances: Dict[CurrencyType, Decimal] = field(default_factory=dict)
    
    # Reputation and limits
    reputation_score: float = 1.0  # 0.0 to 5.0
    spending_limit_usd: Decimal = field(default_factory=lambda: Decimal('1000'))
    credit_limit_usd: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Billing information
    billing_address: Optional[Dict[str, str]] = None
    payment_methods: List[str] = field(default_factory=list)
    default_payment_method: Optional[str] = None
    
    # Account metadata
    account_type: str = "standard"  # standard, premium, enterprise
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Preferences
    auto_recharge_enabled: bool = False
    auto_recharge_threshold: Decimal = field(default_factory=lambda: Decimal('50'))
    auto_recharge_amount: Decimal = field(default_factory=lambda: Decimal('100'))
    
    def get_balance(self, currency: CurrencyType) -> Decimal:
        """Get balance for a specific currency"""
        return self.balances.get(currency, Decimal('0'))
    
    def has_sufficient_balance(self, amount: Decimal, currency: CurrencyType) -> bool:
        """Check if account has sufficient balance"""
        return self.get_balance(currency) >= amount

@dataclass
class ComputeRecommendation:
    """A compute resource recommendation with guidance"""
    recommendation_id: str
    provider_name: str
    instance_type: str
    
    # Basic information
    title: str
    description: str
    use_case_tags: List[str] = field(default_factory=list)
    
    # Pricing information (for guidance only)
    estimated_hourly_cost: Price
    estimated_monthly_cost: Price
    cost_breakdown: Dict[str, Price] = field(default_factory=dict)
    
    # Technical specifications
    specs: Dict[str, Any] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Guidance and instructions
    setup_guide: str = ""
    configuration_steps: List[str] = field(default_factory=list)
    required_credentials: List[str] = field(default_factory=list)
    estimated_setup_time: str = "30 minutes"
    
    # Pros and cons
    advantages: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)
    
    # Community feedback
    community_rating: float = 0.0  # 0.0 to 5.0
    user_reviews_count: int = 0
    expert_verified: bool = False
    
    # Links and resources
    provider_signup_url: str = ""
    documentation_url: str = ""
    tutorial_links: List[str] = field(default_factory=list)
    
    # Metadata
    status: RecommendationStatus = RecommendationStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_price_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ProvisioningGuide:
    """Step-by-step provisioning guidance"""
    guide_id: str
    provider_name: str
    resource_type: str
    
    # Guide content
    title: str
    overview: str
    prerequisites: List[str] = field(default_factory=list)
    
    # Step-by-step instructions
    steps: List[Dict[str, Any]] = field(default_factory=list)  # Each step has title, description, commands, screenshots
    
    # Security and best practices
    security_checklist: List[str] = field(default_factory=list)
    cost_optimization_tips: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    
    # Verification
    verification_steps: List[str] = field(default_factory=list)
    troubleshooting: Dict[str, str] = field(default_factory=dict)
    
    # Community contributions
    contributed_by: str = ""
    community_verified: bool = False
    helpfulness_score: float = 0.0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Transaction:
    """A marketplace transaction"""
    transaction_id: str
    buyer_id: str
    seller_id: str
    listing_id: str
    
    # Transaction details
    quantity: int
    unit_price: Price
    total_amount: Price
    
    # Fees and taxes
    platform_fee: Price
    payment_processing_fee: Price
    taxes: Price = field(default_factory=lambda: Price(Decimal('0'), CurrencyType.USD))
    
    # Status and timestamps
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Payment information
    payment_method_id: Optional[str] = None
    payment_intent_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total_with_fees(self) -> Price:
        """Calculate total amount including all fees"""
        total = self.total_amount.amount + self.platform_fee.amount + \
                self.payment_processing_fee.amount + self.taxes.amount
        return Price(total, self.total_amount.currency, self.total_amount.billing_cycle)

@dataclass
class Review:
    """Review for a marketplace listing"""
    review_id: str
    listing_id: str
    reviewer_id: str
    transaction_id: str
    
    rating: float  # 1.0 to 5.0
    title: str
    content: str
    
    # Verification
    verified_purchase: bool = True
    helpful_votes: int = 0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PricingEngine:
    """Dynamic pricing engine for marketplace resources"""
    
    def __init__(self):
        self.base_rates = {
            ListingType.COMPUTE: Decimal('0.10'),  # Per compute unit hour
            ListingType.STORAGE: Decimal('0.02'),  # Per GB hour
            ListingType.MODEL: Decimal('0.001'),   # Per inference
            ListingType.API_ACCESS: Decimal('0.01'), # Per API call
        }
        
        self.demand_multipliers: Dict[ListingType, float] = {}
        self.supply_multipliers: Dict[ListingType, float] = {}
        
        logger.info("Pricing engine initialized")
    
    async def calculate_dynamic_price(self, listing_type: ListingType, 
                                    demand_factor: float = 1.0,
                                    supply_factor: float = 1.0,
                                    quality_factor: float = 1.0) -> Decimal:
        """Calculate dynamic price based on market conditions"""
        base_rate = self.base_rates.get(listing_type, Decimal('0.01'))
        
        # Apply demand/supply dynamics
        demand_adjustment = 1.0 + (demand_factor - 1.0) * 0.5  # 50% weight
        supply_adjustment = 1.0 / (1.0 + (supply_factor - 1.0) * 0.3)  # 30% weight inverse
        quality_adjustment = 0.5 + quality_factor * 0.5  # 0.5x to 1.0x based on quality
        
        final_price = base_rate * Decimal(str(demand_adjustment * supply_adjustment * quality_adjustment))
        
        # Ensure minimum price
        return max(final_price, Decimal('0.001'))
    
    async def get_market_rates(self) -> Dict[ListingType, Decimal]:
        """Get current market rates for all resource types"""
        rates = {}
        for listing_type in ListingType:
            rates[listing_type] = await self.calculate_dynamic_price(listing_type)
        return rates

class PaymentProcessor:
    """Handles payment processing and billing"""
    
    def __init__(self):
        self.stripe_enabled = STRIPE_AVAILABLE
        if self.stripe_enabled:
            # In production, set from environment
            stripe.api_key = "sk_test_..."
        
        logger.info(f"Payment processor initialized (Stripe: {self.stripe_enabled})")
    
    async def process_payment(self, transaction: Transaction, 
                            payment_method_id: str) -> bool:
        """Process payment for a transaction"""
        start_time = time.time()
        
        try:
            amount_cents = int(transaction.calculate_total_with_fees().amount * 100)
            
            if self.stripe_enabled:
                # Create Stripe payment intent
                payment_intent = stripe.PaymentIntent.create(
                    amount=amount_cents,
                    currency='usd',
                    payment_method=payment_method_id,
                    confirm=True,
                    metadata={
                        'transaction_id': transaction.transaction_id,
                        'listing_id': transaction.listing_id,
                        'buyer_id': transaction.buyer_id
                    }
                )
                
                transaction.payment_intent_id = payment_intent['id']
                success = payment_intent['status'] == 'succeeded'
            else:
                # Mock payment processing
                await asyncio.sleep(0.1)  # Simulate processing time
                success = True
                transaction.payment_intent_id = f"pi_mock_{uuid.uuid4().hex[:8]}"
            
            if success:
                transaction.status = TransactionStatus.COMPLETED
                transaction.completed_at = datetime.now(timezone.utc)
                
                MARKETPLACE_TRANSACTIONS.labels(type="payment", status="success").inc()
                REVENUE_GENERATED.inc(float(transaction.calculate_total_with_fees().amount))
            else:
                transaction.status = TransactionStatus.FAILED
                MARKETPLACE_TRANSACTIONS.labels(type="payment", status="failed").inc()
            
            processing_time = time.time() - start_time
            TRANSACTION_LATENCY.observe(processing_time)
            
            logger.info(f"Payment processing {'succeeded' if success else 'failed'} for transaction {transaction.transaction_id}")
            return success
            
        except Exception as e:
            logger.error(f"Payment processing error for transaction {transaction.transaction_id}: {e}")
            transaction.status = TransactionStatus.FAILED
            MARKETPLACE_TRANSACTIONS.labels(type="payment", status="error").inc()
            return False
    
    async def refund_payment(self, transaction: Transaction, 
                           reason: str = "customer_request") -> bool:
        """Process refund for a transaction"""
        try:
            if transaction.payment_intent_id and self.stripe_enabled:
                # Create Stripe refund
                refund = stripe.Refund.create(
                    payment_intent=transaction.payment_intent_id,
                    reason=reason
                )
                success = refund['status'] == 'succeeded'
            else:
                # Mock refund
                success = True
            
            if success:
                transaction.status = TransactionStatus.REFUNDED
                MARKETPLACE_TRANSACTIONS.labels(type="refund", status="success").inc()
            
            logger.info(f"Refund {'succeeded' if success else 'failed'} for transaction {transaction.transaction_id}")
            return success
            
        except Exception as e:
            logger.error(f"Refund processing error for transaction {transaction.transaction_id}: {e}")
            return False

class MarketplaceAnalytics:
    """Analytics and reporting for marketplace"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
    
    async def generate_revenue_report(self, start_date: datetime, 
                                    end_date: datetime) -> Dict[str, Any]:
        """Generate revenue report for date range"""
        # Mock analytics - in production, query actual transaction database
        total_revenue = Decimal('12459.67')
        transaction_count = 1847
        avg_transaction_value = total_revenue / transaction_count
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "revenue": {
                "total_usd": float(total_revenue),
                "total_transactions": transaction_count,
                "average_transaction_usd": float(avg_transaction_value)
            },
            "breakdown_by_type": {
                "compute": {"revenue": 5234.12, "transactions": 892},
                "models": {"revenue": 3892.45, "transactions": 445},
                "storage": {"revenue": 1876.33, "transactions": 321},
                "services": {"revenue": 1456.77, "transactions": 189}
            },
            "top_performers": [
                {"listing_id": "model_123", "revenue": 1234.56, "sales": 89},
                {"listing_id": "compute_456", "revenue": 987.65, "sales": 67}
            ]
        }
    
    async def calculate_market_metrics(self) -> Dict[str, Any]:
        """Calculate key market metrics"""
        return {
            "total_listings": 1247,
            "active_listings": 892,
            "total_providers": 156,
            "total_buyers": 2341,
            "average_listing_quality": 4.2,
            "market_liquidity": 0.87,  # 0-1 scale
            "price_volatility": 0.23,  # 0-1 scale
            "demand_growth_30d": 0.15,  # 15% growth
            "supply_growth_30d": 0.12   # 12% growth
        }

class ReputationSystem:
    """Reputation and trust scoring system"""
    
    def __init__(self):
        self.reputation_weights = {
            "transaction_success_rate": 0.3,
            "customer_satisfaction": 0.25,
            "delivery_time": 0.2,
            "quality_metrics": 0.15,
            "dispute_rate": -0.1  # Negative weight
        }
    
    async def calculate_reputation_score(self, provider_id: str, 
                                       transactions: List[Transaction],
                                       reviews: List[Review]) -> float:
        """Calculate reputation score for a provider"""
        if not transactions:
            return 1.0  # Neutral score for new providers
        
        # Transaction success rate
        successful_transactions = len([t for t in transactions if t.status == TransactionStatus.COMPLETED])
        success_rate = successful_transactions / len(transactions)
        
        # Customer satisfaction (from reviews)
        if reviews:
            avg_rating = sum(r.rating for r in reviews) / len(reviews)
            satisfaction_score = avg_rating / 5.0  # Normalize to 0-1
        else:
            satisfaction_score = 0.5  # Neutral if no reviews
        
        # Delivery time performance (mock calculation)
        delivery_score = 0.8  # Assume good delivery performance
        
        # Quality metrics (mock)
        quality_score = 0.85
        
        # Dispute rate (mock)
        dispute_rate = 0.02  # 2% dispute rate
        
        # Calculate weighted score
        reputation_score = (
            self.reputation_weights["transaction_success_rate"] * success_rate +
            self.reputation_weights["customer_satisfaction"] * satisfaction_score +
            self.reputation_weights["delivery_time"] * delivery_score +
            self.reputation_weights["quality_metrics"] * quality_score +
            self.reputation_weights["dispute_rate"] * dispute_rate
        )
        
        # Clamp to 0-5 range
        return max(0.0, min(5.0, reputation_score * 5.0))

class ComputeAdvisor:
    """Intelligent compute resource advisor and guidance system"""
    
    def __init__(self):
        self.recommendations: Dict[str, ComputeRecommendation] = {}
        self.guides: Dict[str, ProvisioningGuide] = {}
        self.provider_apis = self._initialize_provider_apis()
        self.cost_calculator = CostCalculator()
        
        logger.info("Compute advisor system initialized")
    
    def _initialize_provider_apis(self) -> Dict[str, Any]:
        """Initialize provider API clients for price discovery (read-only)"""
        return {
            "aws": None,  # boto3 client for pricing API only
            "gcp": None,  # Google Cloud pricing API
            "azure": None,  # Azure pricing API
            "lambda_labs": None,  # Lambda Labs pricing API
            "vast_ai": None,  # Vast.ai pricing API
        }
    
    async def get_compute_recommendations(self, requirements: Dict[str, Any]) -> List[ComputeRecommendation]:
        """Get personalized compute recommendations based on requirements"""
        
        # Parse requirements
        gpu_type = requirements.get("gpu_type", "A100")
        gpu_count = requirements.get("gpu_count", 1)
        max_budget = requirements.get("max_budget_usd", 1000)
        use_case = requirements.get("use_case", "training")
        experience_level = requirements.get("experience_level", "beginner")
        
        recommendations = []
        
        # AWS Recommendation
        aws_rec = await self._create_aws_recommendation(gpu_type, gpu_count, max_budget, experience_level)
        if aws_rec:
            recommendations.append(aws_rec)
        
        # GCP Recommendation
        gcp_rec = await self._create_gcp_recommendation(gpu_type, gpu_count, max_budget, experience_level)
        if gcp_rec:
            recommendations.append(gcp_rec)
        
        # Community providers
        community_recs = await self._get_community_recommendations(gpu_type, gpu_count, max_budget)
        recommendations.extend(community_recs)
        
        # Sort by user preference (cost, ease of use, performance)
        recommendations.sort(key=lambda x: (x.estimated_hourly_cost.amount, -x.community_rating))
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _create_aws_recommendation(self, gpu_type: str, gpu_count: int, max_budget: float, experience_level: str) -> Optional[ComputeRecommendation]:
        """Create AWS recommendation with setup guidance"""
        
        instance_map = {
            "A100": "p4d.24xlarge" if gpu_count >= 8 else "p4de.24xlarge",
            "V100": "p3.2xlarge" if gpu_count == 1 else "p3.8xlarge",
            "H100": "p5.48xlarge"
        }
        
        instance_type = instance_map.get(gpu_type, "p3.2xlarge")
        hourly_cost = self._get_aws_pricing(instance_type)
        
        if hourly_cost * 100 > max_budget:  # Assume 100 hours max
            return None
        
        setup_complexity = "Intermediate" if experience_level == "beginner" else "Easy"
        
        return ComputeRecommendation(
            recommendation_id=f"aws-{instance_type}-{int(time.time())}",
            provider_name="Amazon Web Services (AWS)",
            instance_type=instance_type,
            title=f"AWS {instance_type} - Professional Cloud GPU",
            description=f"Industry-standard {gpu_type} GPU on AWS with enterprise support",
            use_case_tags=["professional", "enterprise", "scalable"],
            estimated_hourly_cost=Price(Decimal(str(hourly_cost)), CurrencyType.USD),
            estimated_monthly_cost=Price(Decimal(str(hourly_cost * 730)), CurrencyType.USD),
            specs={
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "gpu_memory": "40GB" if gpu_type == "A100" else "16GB",
                "cpu_cores": 96,
                "ram_gb": 1152,
                "storage": "8x 1.9TB NVMe SSD",
                "network": "100 Gbps"
            },
            setup_guide=f"Setting up AWS {instance_type} requires an AWS account and basic cloud knowledge.",
            configuration_steps=[
                "1. Create AWS account and verify payment method",
                "2. Request GPU quota increase (may take 24-48 hours)",
                "3. Create EC2 key pair for SSH access",
                "4. Launch instance with Deep Learning AMI",
                "5. Configure security groups for SSH/Jupyter access",
                "6. Install your training framework and data",
                "7. Monitor costs with CloudWatch billing alerts"
            ],
            required_credentials=["AWS Account", "Credit Card", "Government ID (for quota)"],
            estimated_setup_time="2-4 hours (plus quota approval wait)",
            advantages=[
                "Enterprise-grade reliability and support",
                "Extensive documentation and tutorials",
                "Integration with other AWS services",
                "Global availability zones",
                "Pay-per-second billing"
            ],
            limitations=[
                "Requires quota approval for GPU instances",
                "Can be expensive for long-term usage",
                "Complex pricing structure",
                "Learning curve for AWS console"
            ],
            best_for=[
                "Production training workloads",
                "Teams already using AWS",
                "Need for enterprise support",
                "Compliance requirements"
            ],
            provider_signup_url="https://aws.amazon.com/ec2/instance-types/p4/",
            documentation_url="https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html",
            tutorial_links=[
                "https://aws.amazon.com/getting-started/hands-on/train-deep-learning-model-aws-ec2-containers/",
                "https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html"
            ],
            community_rating=4.2,
            user_reviews_count=1547,
            expert_verified=True,
            status=RecommendationStatus.VERIFIED
        )
    
    def _get_aws_pricing(self, instance_type: str) -> float:
        """Get current AWS pricing (read-only, no provisioning)"""
        # Mock pricing - in real implementation, use AWS Pricing API
        pricing_map = {
            "p4d.24xlarge": 32.77,
            "p4de.24xlarge": 40.96,
            "p3.2xlarge": 3.06,
            "p3.8xlarge": 12.24,
            "p5.48xlarge": 98.32
        }
        return pricing_map.get(instance_type, 10.0)
    
    async def create_setup_guide(self, provider: str, resource_type: str) -> ProvisioningGuide:
        """Create comprehensive setup guide for a provider/resource"""
        
        if provider.lower() == "aws":
            return ProvisioningGuide(
                guide_id=f"aws-{resource_type}-guide",
                provider_name="Amazon Web Services",
                resource_type=resource_type,
                title=f"Complete AWS {resource_type} Setup Guide",
                overview=f"Step-by-step guide to set up {resource_type} on AWS safely and cost-effectively.",
                prerequisites=[
                    "Valid email address and phone number",
                    "Credit card or debit card",
                    "Government-issued ID (for account verification)",
                    "Basic command line knowledge"
                ],
                steps=[
                    {
                        "title": "Create AWS Account",
                        "description": "Sign up for AWS and verify your account",
                        "commands": [],
                        "screenshots": ["aws-signup.png"],
                        "estimated_time": "15 minutes",
                        "cost_impact": "$0 (free tier eligible)"
                    },
                    {
                        "title": "Request GPU Quota",
                        "description": "Request quota increase for GPU instances",
                        "commands": [],
                        "screenshots": ["quota-request.png"],
                        "estimated_time": "5 minutes (24-48 hour approval)",
                        "cost_impact": "$0"
                    },
                    {
                        "title": "Set Up Billing Alerts",
                        "description": "Configure billing alerts to avoid surprise costs",
                        "commands": [],
                        "screenshots": ["billing-alerts.png"],
                        "estimated_time": "10 minutes",
                        "cost_impact": "$0"
                    },
                    {
                        "title": "Launch GPU Instance",
                        "description": "Launch your first GPU instance with recommended settings",
                        "commands": [
                            "# Use AWS CLI to launch instance",
                            "aws ec2 run-instances --image-id ami-0c02fb55956c7d316 --instance-type p3.2xlarge"
                        ],
                        "screenshots": ["launch-instance.png"],
                        "estimated_time": "20 minutes",
                        "cost_impact": "$3.06/hour"
                    }
                ],
                security_checklist=[
                    "Enable MFA on your AWS account",
                    "Use IAM roles instead of root account",
                    "Configure security groups to restrict SSH access",
                    "Enable CloudTrail for audit logging",
                    "Set up billing alerts to monitor costs"
                ],
                cost_optimization_tips=[
                    "Use Spot Instances for 50-90% cost savings",
                    "Set up auto-shutdown scripts to avoid idle charges",
                    "Use S3 for data storage instead of EBS when possible",
                    "Monitor usage with Cost Explorer",
                    "Consider Reserved Instances for predictable workloads"
                ],
                common_pitfalls=[
                    "Forgetting to stop instances after use",
                    "Not requesting GPU quota increase in advance",
                    "Using the wrong instance type for your workload",
                    "Not setting up proper monitoring and alerts",
                    "Storing large datasets on expensive EBS volumes"
                ],
                verification_steps=[
                    "SSH into the instance successfully",
                    "Verify GPU is detected: nvidia-smi",
                    "Test basic PyTorch GPU: torch.cuda.is_available()",
                    "Check billing dashboard shows expected charges"
                ],
                troubleshooting={
                    "SSH connection refused": "Check security group allows SSH from your IP",
                    "GPU not detected": "Ensure you selected a GPU instance type",
                    "Out of capacity error": "Try different availability zone",
                    "High costs": "Check if instances are running when not needed"
                },
                contributed_by="LoomOS Team",
                community_verified=True,
                helpfulness_score=4.8
            )
        
        # Default generic guide
        return ProvisioningGuide(
            guide_id=f"{provider}-{resource_type}-guide",
            provider_name=provider,
            resource_type=resource_type,
            title=f"{provider} {resource_type} Setup Guide",
            overview=f"Basic setup guide for {provider} {resource_type}."
        )
    
    async def estimate_costs(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed cost estimation and budgeting guidance"""
        
        gpu_hours = requirements.get("training_hours", 24)
        gpu_type = requirements.get("gpu_type", "A100")
        gpu_count = requirements.get("gpu_count", 1)
        
        estimates = {
            "aws": {
                "hourly": 32.77 * gpu_count,
                "total": 32.77 * gpu_count * gpu_hours,
                "breakdown": {
                    "compute": 32.77 * gpu_count * gpu_hours,
                    "storage": 0.10 * gpu_hours,  # EBS
                    "data_transfer": 0.09 * 100,  # Assume 100GB
                }
            },
            "gcp": {
                "hourly": 29.15 * gpu_count,
                "total": 29.15 * gpu_count * gpu_hours,
                "breakdown": {
                    "compute": 29.15 * gpu_count * gpu_hours,
                    "storage": 0.08 * gpu_hours,
                    "data_transfer": 0.08 * 100,
                }
            },
            "budget_recommendations": {
                "add_buffer": "Add 20-30% buffer for unexpected costs",
                "monitoring": "Set billing alerts at 50%, 80%, and 100% of budget",
                "optimization": "Consider spot instances for 50-70% savings",
                "data_costs": "Factor in data transfer and storage costs"
            }
        }
        
        return estimates
    
    async def get_user_guidance(self, experience_level: str, budget: float) -> Dict[str, Any]:
        """Provide personalized guidance based on user profile"""
        
        if experience_level == "beginner":
            return {
                "recommended_approach": "Start small with single GPU instances",
                "suggested_providers": ["Google Colab Pro", "Lambda Labs", "Vast.ai"],
                "learning_path": [
                    "1. Try Google Colab Pro ($10/month) to learn basics",
                    "2. Experiment with Vast.ai for affordable GPU access", 
                    "3. Move to AWS/GCP when you need production features"
                ],
                "budget_guidance": "Start with $50-100/month learning budget",
                "safety_tips": [
                    "Always set billing alerts",
                    "Start with time-limited instances",
                    "Use community tutorials and guides",
                    "Join Discord/forums for help"
                ]
            }
        elif experience_level == "intermediate":
            return {
                "recommended_approach": "Use managed services or spot instances",
                "suggested_providers": ["AWS SageMaker", "Google AI Platform", "Azure ML"],
                "optimization_tips": [
                    "Use spot instances for training",
                    "Implement auto-shutdown scripts",
                    "Optimize data pipeline for faster training",
                    "Use mixed precision training"
                ],
                "budget_guidance": f"Budget ${budget * 1.2:.0f} with 20% buffer",
                "scaling_advice": "Start single-node, scale to multi-node when needed"
            }
        else:  # expert
            return {
                "recommended_approach": "Multi-cloud strategy with cost optimization",
                "advanced_features": ["Reserved instances", "Custom AMIs", "Auto-scaling"],
                "cost_optimization": [
                    "Implement intelligent workload scheduling",
                    "Use preemptible/spot instances aggressively", 
                    "Consider bare metal for large workloads",
                    "Implement cross-cloud arbitrage"
                ],
                "monitoring": "Set up comprehensive cost tracking and anomaly detection"
            }
        """Debit an account"""
        if user_id not in self.accounts:
            logger.error(f"Account {user_id} not found")
            return False
        
        account = self.accounts[user_id]
        
        if not account.has_sufficient_balance(amount, currency):
            logger.error(f"Insufficient balance for user {user_id}")
            return False
        
        current_balance = account.get_balance(currency)
        account.balances[currency] = current_balance - amount
        account.last_activity = datetime.now(timezone.utc)
        
        logger.info(f"Debited {amount} {currency.value} from account {user_id}")
        return True
    
    async def create_listing(self, provider_id: str, listing_type: ListingType,
                           title: str, description: str, price: Price,
                           **kwargs) -> str:
        """Create a new marketplace listing"""
        listing_id = f"{listing_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Calculate quality score based on provider reputation
        provider_reputation = await self._get_provider_reputation(provider_id)
        quality_score = min(5.0, provider_reputation + 0.5)  # Boost for listings
        
        listing = MarketplaceListing(
            listing_id=listing_id,
            provider_id=provider_id,
            listing_type=listing_type,
            title=title,
            description=description,
            price=price,
            quality_score=quality_score,
            **kwargs
        )
        
        self.listings[listing_id] = listing
        ACTIVE_LISTINGS.inc()
        
        logger.info(f"Created listing {listing_id} by provider {provider_id}")
        return listing_id
    
    async def search_listings(self, listing_type: Optional[ListingType] = None,
                            max_price: Optional[Price] = None,
                            min_quality: float = 0.0,
                            tags: List[str] = None) -> List[MarketplaceListing]:
        """Search marketplace listings"""
        results = []
        
        for listing in self.listings.values():
            if listing.status != ListingStatus.ACTIVE:
                continue
            
            # Type filter
            if listing_type and listing.listing_type != listing_type:
                continue
            
            # Price filter
            if max_price and listing.price.to_usd(self.exchange_rates) > max_price.to_usd(self.exchange_rates):
                continue
            
            # Quality filter
            if listing.quality_score < min_quality:
                continue
            
            # Tags filter
            if tags and not any(tag in listing.tags for tag in tags):
                continue
            
            results.append(listing)
        
        # Sort by quality score and relevance
        results.sort(key=lambda l: (l.quality_score, l.total_sales), reverse=True)
        
        logger.info(f"Found {len(results)} listings matching search criteria")
        return results
    
    async def purchase_listing(self, buyer_id: str, listing_id: str, 
                             quantity: int = 1,
                             payment_method_id: Optional[str] = None) -> str:
        """Purchase a marketplace listing"""
        if listing_id not in self.listings:
            raise ValueError(f"Listing {listing_id} not found")
        
        listing = self.listings[listing_id]
        
        if listing.status != ListingStatus.ACTIVE:
            raise ValueError(f"Listing {listing_id} is not available")
        
        if listing.available_quantity and quantity > listing.available_quantity:
            raise ValueError(f"Insufficient quantity available")
        
        # Create transaction
        transaction_id = f"txn_{uuid.uuid4().hex[:8]}"
        unit_price = listing.price
        total_amount = Price(
            unit_price.amount * quantity,
            unit_price.currency,
            unit_price.billing_cycle
        )
        
        # Calculate fees
        platform_fee_rate = Decimal('0.05')  # 5% platform fee
        payment_fee_rate = Decimal('0.029')  # 2.9% payment processing fee
        
        platform_fee = Price(
            total_amount.amount * platform_fee_rate,
            total_amount.currency
        )
        
        payment_fee = Price(
            total_amount.amount * payment_fee_rate,
            total_amount.currency
        )
        
        transaction = Transaction(
            transaction_id=transaction_id,
            buyer_id=buyer_id,
            seller_id=listing.provider_id,
            listing_id=listing_id,
            quantity=quantity,
            unit_price=unit_price,
            total_amount=total_amount,
            platform_fee=platform_fee,
            payment_processing_fee=payment_fee,
            payment_method_id=payment_method_id
        )
        
        self.transactions[transaction_id] = transaction
        
        # Process payment
        if payment_method_id:
            payment_success = await self.payment_processor.process_payment(
                transaction, payment_method_id
            )
        else:
            # Use account balance
            total_with_fees = transaction.calculate_total_with_fees()
            payment_success = await self.debit_account(
                buyer_id, total_with_fees.amount, total_with_fees.currency
            )
            
            if payment_success:
                transaction.status = TransactionStatus.COMPLETED
                transaction.completed_at = datetime.now(timezone.utc)
        
        if payment_success:
            # Update listing
            if listing.available_quantity:
                listing.available_quantity -= quantity
                if listing.available_quantity <= 0:
                    listing.status = ListingStatus.SOLD_OUT
            
            listing.total_sales += quantity
            listing.total_revenue += total_amount.amount
            
            # Credit seller account
            seller_amount = total_amount.amount - platform_fee.amount
            await self.credit_account(listing.provider_id, seller_amount, total_amount.currency)
            
            logger.info(f"Purchase completed: {transaction_id}")
        else:
            logger.error(f"Purchase failed: {transaction_id}")
        
        return transaction_id
    
    async def add_review(self, reviewer_id: str, listing_id: str, 
                        transaction_id: str, rating: float, 
                        title: str, content: str) -> str:
        """Add a review for a listing"""
        review_id = f"review_{uuid.uuid4().hex[:8]}"
        
        review = Review(
            review_id=review_id,
            listing_id=listing_id,
            reviewer_id=reviewer_id,
            transaction_id=transaction_id,
            rating=max(1.0, min(5.0, rating)),  # Clamp to 1-5
            title=title,
            content=content
        )
        
        self.reviews[review_id] = review
        
        # Update listing rating
        listing = self.listings[listing_id]
        listing_reviews = [r for r in self.reviews.values() if r.listing_id == listing_id]
        
        if listing_reviews:
            avg_rating = sum(r.rating for r in listing_reviews) / len(listing_reviews)
            listing.average_rating = avg_rating
            listing.review_count = len(listing_reviews)
        
        logger.info(f"Added review {review_id} for listing {listing_id}")
        return review_id
    
    async def _get_provider_reputation(self, provider_id: str) -> float:
        """Get reputation score for a provider"""
        provider_transactions = [t for t in self.transactions.values() if t.seller_id == provider_id]
        provider_reviews = [r for r in self.reviews.values() 
                          if self.listings.get(r.listing_id, {}).provider_id == provider_id]
        
        if not provider_transactions:
            return 1.0  # Neutral score for new providers
        
        return await self.reputation_system.calculate_reputation_score(
            provider_id, provider_transactions, provider_reviews
        )
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        active_listings = len([l for l in self.listings.values() if l.status == ListingStatus.ACTIVE])
        total_transactions = len(self.transactions)
        total_revenue = sum(t.total_amount.amount for t in self.transactions.values() 
                          if t.status == TransactionStatus.COMPLETED)
        
        return {
            "total_accounts": len(self.accounts),
            "total_listings": len(self.listings),
            "active_listings": active_listings,
            "total_transactions": total_transactions,
            "total_revenue_usd": float(total_revenue),
            "average_transaction_value": float(total_revenue / max(total_transactions, 1)),
            "total_reviews": len(self.reviews),
            "average_review_rating": sum(r.rating for r in self.reviews.values()) / max(len(self.reviews), 1),
            "marketplace_health": "healthy" if active_listings > 0 else "low_activity"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test account operations
            test_user = f"health_test_{uuid.uuid4().hex[:8]}"
            test_account = await self.create_account(test_user, f"{test_user}@test.com")
            credit_success = await self.credit_account(test_user, Decimal('10'), CurrencyType.USD)
            
            # Test listing creation
            test_listing = await self.create_listing(
                test_user, ListingType.SERVICE, "Health Check Test", "Test listing",
                Price(Decimal('1.0'), CurrencyType.USD)
            )
            
            # Cleanup
            if test_user in self.accounts:
                del self.accounts[test_user]
            if test_listing in self.listings:
                del self.listings[test_listing]
            
            stats = await self.get_marketplace_stats()
            
            return {
                "status": "healthy",
                "account_creation": test_account is not None,
                "credit_operation": credit_success,
                "listing_creation": test_listing is not None,
                "marketplace_stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Marketplace health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Factory function

def create_compute_advisor() -> ComputeAdvisor:
    """Create a new compute advisor instance"""
    return ComputeAdvisor()

# Example usage
async def example_advisory_session():
    """Example of how the advisory system works"""
    advisor = create_compute_advisor()
    
    # User provides their requirements
    user_requirements = {
        "gpu_type": "A100",
        "gpu_count": 4,
        "training_hours": 48,
        "max_budget_usd": 5000,
        "use_case": "fine_tuning_llm",
        "experience_level": "beginner",
        "priority": "cost_effective"
    }
    
    print("🎯 LoomOS Compute Advisory Session")
    print("="*50)
    
    # Get personalized recommendations
    recommendations = await advisor.get_compute_recommendations(user_requirements)
    
    print(f"\n📊 Found {len(recommendations)} recommendations for your needs:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.title}")
        print(f"   💰 Cost: ${rec.estimated_hourly_cost.amount}/hour")
        print(f"   ⭐ Rating: {rec.community_rating}/5.0 ({rec.user_reviews_count} reviews)")
        print(f"   🛠️  Setup: {rec.estimated_setup_time}")
        print(f"   📋 Best for: {', '.join(rec.best_for[:2])}")
    
    # Get detailed setup guide
    setup_guide = await advisor.create_setup_guide("aws", "gpu_instance")
    print(f"\n📚 Setup Guide: {setup_guide.title}")
    print(f"   ⏱️  Prerequisites: {len(setup_guide.prerequisites)} items")
    print(f"   📝 Steps: {len(setup_guide.steps)} detailed steps")
    print(f"   🛡️  Security items: {len(setup_guide.security_checklist)}")
    
    # Get cost estimates
    cost_estimates = await advisor.estimate_costs(user_requirements)
    print(f"\n💰 Cost Estimates:")
    print(f"   AWS: ${cost_estimates['aws']['total']:.2f} total")
    print(f"   GCP: ${cost_estimates['gcp']['total']:.2f} total")
    
    # Get personalized guidance
    guidance = await advisor.get_user_guidance("beginner", 5000)
    print(f"\n🎓 Personalized Guidance:")
    print(f"   Approach: {guidance['recommended_approach']}")
    print(f"   Budget: {guidance['budget_guidance']}")
    print(f"   Safety: {len(guidance['safety_tips'])} safety tips")
    
    print("\n✅ Advisory session complete!")
    print("Next steps: Review recommendations and follow setup guides")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_advisory_session())

# Legacy compatibility
async def credit_account_legacy(user: str, amount: float) -> None:
    """Legacy credit_account function"""
    marketplace = create_marketplace()
    await marketplace.credit_account(user, Decimal(str(amount)))