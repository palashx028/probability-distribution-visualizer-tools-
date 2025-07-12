import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    bernoulli, binom, poisson, geom, nbinom, hypergeom, randint, multinomial,
    norm, uniform, expon, gamma, beta, weibull_min, chi2, t, f, lognorm,
    skew, kurtosis
)

#  Formulas for reference (Latex-style text)
distribution_formulas = {
    "Bernoulli": "P(X = x) = p^x * (1 - p)^(1 - x), x ∈ {0,1}",
    "Binomial": "P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)",
    "Poisson": "P(X = k) = (λ^k * e^(-λ)) / k!",
    "Geometric": "P(X = k) = (1 - p)^(k - 1) * p",
    "Negative Binomial": "P(X = k) = C(k - 1, r - 1) * p^r * (1 - p)^(k - r)",
    "Hypergeometric": "P(X = k) = [C(K, k) * C(N - K, n - k)] / C(N, n)",
    "Discrete Uniform": "P(X = x) = 1 / (b - a + 1)",
    "Multinomial": "P(X_1 = x_1, ..., X_k = x_k) = n! / (x1!...xk!) * p1^x1 * ... * pk^xk",
    "Normal": "f(x) = (1 / √(2πσ²)) * e^(-(x - μ)² / (2σ²))",
    "Continuous Uniform": "f(x) = 1 / (b - a) for a ≤ x ≤ b",
    "Exponential": "f(x) = λ * e^(-λx) for x ≥ 0",
    "Gamma": "f(x) = (x^(k - 1) * e^(-x/θ)) / (Γ(k) * θ^k)",
    "Beta": "f(x) = (x^{α-1} * (1 - x)^{β - 1}) / B(α, β)",
    "Weibull": "f(x) = (c / λ) * (x / λ)^(c - 1) * e^(-(x / λ)^c)",
    "Chi-Square": "f(x) = (1 / (2^{k/2} * Γ(k/2))) * x^{(k/2 - 1)} * e^{-x/2}",
    "Student’s t": "f(t) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + t²/ν)^(-(ν+1)/2)",
    "F": "f(x) = [(d1/d2)^(d1/2) * x^{(d1/2 - 1)}] / B(d1/2, d2/2) * (1 + (d1/d2)x)^(-(d1+d2)/2)",
    "Log-Normal": "f(x) = (1 / (xσ√(2π))) * e^(-(ln(x) - μ)² / (2σ²))"
}

def plot_distribution(data, dist_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, stat="density", bins=30, color="skyblue", edgecolor="black")
    plt.title(f"{dist_name} Distribution\nSkewness: {skew(data):.2f}, Kurtosis: {kurtosis(data):.2f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_formula(dist_name):
    formula = distribution_formulas.get(dist_name, " Formula not available.")
    print(f"\n Distribution Function for {dist_name}:\n{formula}\n")

def generate_distribution(choice):
    size = int(input("Sample size: "))
    if choice == "1":
        p = float(input("Probability of success (p): "))
        return "Bernoulli", bernoulli.rvs(p, size=size)
    elif choice == "2":
        n = int(input("Number of trials (n): "))
        p = float(input("Probability (p): "))
        return "Binomial", binom.rvs(n, p, size=size)
    elif choice == "3":
        lam = float(input("Lambda (λ): "))
        return "Poisson", poisson.rvs(mu=lam, size=size)
    elif choice == "4":
        p = float(input("Probability (p): "))
        return "Geometric", geom.rvs(p, size=size)
    elif choice == "5":
        r = int(input("Successes (r): "))
        p = float(input("Probability (p): "))
        return "Negative Binomial", nbinom.rvs(r, p, size=size)
    elif choice == "6":
        M = int(input("Population (M): "))
        n = int(input("Success states (n): "))
        N = int(input("Number of draws (N): "))
        return "Hypergeometric", hypergeom.rvs(M, n, N, size=size)
    elif choice == "7":
        a = int(input("Lower bound (a): "))
        b = int(input("Upper bound (b): "))
        return "Discrete Uniform", randint.rvs(a, b + 1, size=size)
    elif choice == "8":
        n = int(input("Number of trials: "))
        k = int(input("Number of categories: "))
        probs = list(map(float, input("Enter probabilities (sum=1): ").split()))
        return "Multinomial", multinomial.rvs(n, probs, size=1)[0]
    elif choice == "9":
        mu = float(input("Mean (μ): "))
        sigma = float(input("Std Dev (σ): "))
        return "Normal", norm.rvs(loc=mu, scale=sigma, size=size)
    elif choice == "10":
        a = float(input("Lower bound (a): "))
        b = float(input("Upper bound (b): "))
        return "Continuous Uniform", uniform.rvs(loc=a, scale=b-a, size=size)
    elif choice == "11":
        scale = float(input("Scale (1/λ): "))
        return "Exponential", expon.rvs(scale=scale, size=size)
    elif choice == "12":
        shape = float(input("Shape (k): "))
        scale = float(input("Scale (θ): "))
        return "Gamma", gamma.rvs(shape, scale=scale, size=size)
    elif choice == "13":
        a = float(input("Alpha (α): "))
        b = float(input("Beta (β): "))
        return "Beta", beta.rvs(a, b, size=size)
    elif choice == "14":
        c = float(input("Shape parameter (c): "))
        return "Weibull", weibull_min.rvs(c, size=size)
    elif choice == "15":
        df = float(input("Degrees of freedom: "))
        return "Chi-Square", chi2.rvs(df, size=size)
    elif choice == "16":
        df = float(input("Degrees of freedom: "))
        return "Student’s t", t.rvs(df, size=size)
    elif choice == "17":
        dfn = float(input("Numerator df: "))
        dfd = float(input("Denominator df: "))
        return "F", f.rvs(dfn, dfd, size=size)
    elif choice == "18":
        mean = float(input("Mean of log: "))
        sigma = float(input("Std dev of log: "))
        return "Log-Normal", lognorm.rvs(s=sigma, scale=np.exp(mean), size=size)
    else:
        raise ValueError("Invalid choice.")

def show_menu():
    print("\n Probability Distributions Visualizer")
    print("Select a distribution:")
    print("1. Bernoulli")
    print("2. Binomial")
    print("3. Poisson")
    print("4. Geometric")
    print("5. Negative Binomial")
    print("6. Hypergeometric")
    print("7. Discrete Uniform")
    print("8. Multinomial")
    print("9. Normal")
    print("10. Continuous Uniform")
    print("11. Exponential")
    print("12. Gamma")
    print("13. Beta")
    print("14. Weibull")
    print("15. Chi-Square")
    print("16. Student’s t")
    print("17. F Distribution")
    print("18. Log-Normal")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (0-18): ").strip()
        if choice == "0":
            print(" Exiting program. Goodbye!")
            break
        try:
            dist_name, data = generate_distribution(choice)
            show_formula(dist_name)
            plot_distribution(data, dist_name)
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()
