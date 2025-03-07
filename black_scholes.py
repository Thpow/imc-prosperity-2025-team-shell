import numpy as np
from scipy.stats import norm

class BlackScholes:
    @staticmethod
    def d1(S, K, T, r, sigma):
        """
        Calculate d1 parameter for Black-Scholes formula
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility of the stock (annual)
        """
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter for Black-Scholes formula"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """
        Calculate call option price using Black-Scholes formula
        
        Returns:
        float: Theoretical call option price
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """
        Calculate put option price using Black-Scholes formula
        
        Returns:
        float: Theoretical put option price
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    @staticmethod
    def call_delta(S, K, T, r, sigma):
        """Calculate delta of a call option"""
        return norm.cdf(BlackScholes.d1(S, K, T, r, sigma))
    
    @staticmethod
    def put_delta(S, K, T, r, sigma):
        """Calculate delta of a put option"""
        return -norm.cdf(-BlackScholes.d1(S, K, T, r, sigma))
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate gamma (same for call and put)"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate vega (same for call and put)"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S*np.sqrt(T)*norm.pdf(d1)
    
    @staticmethod
    def call_theta(S, K, T, r, sigma):
        """Calculate theta for a call option"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
        term2 = -r*K*np.exp(-r*T)*norm.cdf(d2)
        return term1 + term2
    
    @staticmethod
    def put_theta(S, K, T, r, sigma):
        """Calculate theta for a put option"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
        term2 = r*K*np.exp(-r*T)*norm.cdf(-d2)
        return term1 + term2
    
    @staticmethod
    def call_rho(S, K, T, r, sigma):
        """Calculate rho for a call option"""
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K*T*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_rho(S, K, T, r, sigma):
        """Calculate rho for a put option"""
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return -K*T*np.exp(-r*T)*norm.cdf(-d2)
