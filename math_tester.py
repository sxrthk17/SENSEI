import math
import numpy as np
import math_solver as m


passed = 0
failed = 0


def check(name, condition):
    global passed, failed

    if condition:
        print(f"✅ PASS — {name}")
        passed += 1
    else:
        print(f"❌ FAIL — {name}")
        failed += 1


def close(a, b, tolerance=1e-6):
    return np.allclose(a, b, rtol=tolerance, atol=tolerance)


print("=" * 70)
print("SENSEI — L5 MASTER INTEGRATION TEST")
print("=" * 70)


# ============================================================
# L5.1 — DESCRIPTIVE STATISTICS
# ============================================================

data = [1, 2, 3, 4, 5]

r = m.calculate_mean(data)
check("Mean", r["success"] and r["result"] == 3)

r = m.calculate_median(data)
check("Median", r["success"] and r["result"] == 3)

r = m.calculate_mode([1, 2, 2, 3])
check("Mode", r["success"] and r["result"] == 2)

r = m.calculate_variance(data)
check("Population variance", r["success"] and close(r["result"], 2))

r = m.calculate_variance(data, sample=True)
check("Sample variance", r["success"] and close(r["result"], 2.5))

r = m.calculate_standard_deviation(data)
check(
    "Population standard deviation",
    r["success"] and close(r["result"], np.sqrt(2))
)

r = m.calculate_percentile(data, 50)
check("50th percentile", r["success"] and r["result"] == 3)

r = m.calculate_covariance(data, [2, 4, 6, 8, 10])
check(
    "Covariance",
    r["success"] and r["result"].shape == (2, 2)
)


# ============================================================
# L5.2 — BASIC PROBABILITY
# ============================================================

r = m.calculate_probability(2, 5)
check("Basic probability", r["success"] and r["result"] == 0.4)


# ============================================================
# L5.2 — CONDITIONAL PROBABILITY
# ============================================================

r = m.calculate_conditional_probability(0.2, 0.5)
check(
    "Conditional probability",
    r["success"] and close(r["result"], 0.4)
)


# ============================================================
# L5.3 — BAYES
# ============================================================

r = m.calculate_bayes_probability(0.2, 0.85, 0.1)
check(
    "Bayes probability",
    r["success"] and close(r["result"], 0.68)
)


# ============================================================
# L5.4 — PMF / PDF / CDF
# ============================================================

r = m.calculate_pmf([0.2, 0.3, 0.5])
check(
    "PMF",
    r["success"] and close(np.sum(r["result"]), 1)
)

r = m.calculate_pdf([0, 1, 2])
check(
    "Normal PDF",
    r["success"] and len(r["result"]) == 3
)

r = m.calculate_cdf([0, 1, 2])
check(
    "Normal CDF",
    r["success"]
    and np.all(r["result"] >= 0)
    and np.all(r["result"] <= 1)
)


# ============================================================
# L5.5 — UNIFORM
# ============================================================

r = m.uniform_pdf([0, 0.5, 1], 0, 1)
check("Uniform PDF", r["success"])

r = m.uniform_cdf([0, 0.5, 1], 0, 1)
check(
    "Uniform CDF",
    r["success"] and close(r["result"], [0, 0.5, 1])
)

r = m.uniform_probability(0.2, 0.6, 0, 1)
check(
    "Uniform probability",
    r["success"] and close(r["result"], 0.4)
)

r = m.uniform_mean(0, 10)
check("Uniform mean", r["success"] and r["result"] == 5)

r = m.uniform_variance(0, 10)
check(
    "Uniform variance",
    r["success"] and close(r["result"], 100 / 12)
)

r = m.plot_uniform_distribution(0, 1)
check("Uniform visualization", r["success"])


# ============================================================
# L5.6 — BERNOULLI
# ============================================================

r = m.bernoulli_pmf([0, 1], 0.5)
check(
    "Bernoulli PMF",
    r["success"] and close(r["result"], [0.5, 0.5])
)

r = m.bernoulli_cdf([0, 1], 0.5)
check("Bernoulli CDF", r["success"])

r = m.bernoulli_probability(1, 0.7)
check(
    "Bernoulli probability",
    r["success"] and close(r["result"], 0.7)
)

r = m.bernoulli_mean(0.7)
check("Bernoulli mean", r["success"] and r["result"] == 0.7)

r = m.bernoulli_variance(0.7)
check(
    "Bernoulli variance",
    r["success"] and close(r["result"], 0.21)
)

r = m.plot_bernoulli_distribution(0.5)
check("Bernoulli visualization", r["success"])


# ============================================================
# L5.6 — BINOMIAL
# ============================================================

r = m.binomial_pmf([0, 5, 10], 10, 0.5)
check("Binomial PMF", r["success"])

r = m.binomial_cdf([0, 5, 10], 10, 0.5)
check("Binomial CDF", r["success"])

r = m.binomial_probability(5, 10, 0.5)
check(
    "Binomial probability",
    r["success"] and close(r["result"], 0.24609375)
)

r = m.binomial_mean(10, 0.5)
check("Binomial mean", r["success"] and r["result"] == 5)

r = m.binomial_variance(10, 0.5)
check("Binomial variance", r["success"] and r["result"] == 2.5)

r = m.plot_binomial_distribution(10, 0.5)
check("Binomial visualization", r["success"])


# ============================================================
# L5.7 — GEOMETRIC
# ============================================================

r = m.geometric_pmf([1, 2, 3], 0.5)
check(
    "Geometric PMF",
    r["success"] and close(r["result"], [0.5, 0.25, 0.125])
)

r = m.geometric_cdf([1, 2, 3], 0.5)
check("Geometric CDF", r["success"])

r = m.geometric_probability(3, 0.5)
check(
    "Geometric probability",
    r["success"] and close(r["result"], 0.125)
)

r = m.geometric_mean(0.5)
check("Geometric mean", r["success"] and r["result"] == 2)

r = m.geometric_variance(0.5)
check("Geometric variance", r["success"] and r["result"] == 2)

r = m.plot_geometric_distribution(0.5)
check("Geometric visualization", r["success"])


# ============================================================
# L5.7 — POISSON
# ============================================================

r = m.poisson_pmf([0, 1, 2], 3)
check("Poisson PMF", r["success"])

r = m.poisson_cdf([0, 1, 2], 3)
check("Poisson CDF", r["success"])

r = m.poisson_probability(4, 3)
check(
    "Poisson probability",
    r["success"] and close(r["result"], 0.1680313557)
)

r = m.poisson_mean(3)
check("Poisson mean", r["success"] and r["result"] == 3)

r = m.poisson_variance(3)
check("Poisson variance", r["success"] and r["result"] == 3)

r = m.plot_poisson_distribution(3)
check("Poisson visualization", r["success"])


# ============================================================
# L5.8 — EXPONENTIAL
# ============================================================

r = m.exponential_pdf([0, 1, 2], 1)
check("Exponential PDF", r["success"])

r = m.exponential_cdf([0, 1, 2], 1)
check("Exponential CDF", r["success"])

r = m.exponential_probability(1, 2, 1)
check(
    "Exponential interval probability",
    r["success"] and close(r["result"], 0.2325441579)
)

r = m.exponential_mean(2)
check(
    "Exponential mean",
    r["success"] and close(r["result"], 0.5)
)

r = m.exponential_variance(2)
check(
    "Exponential variance",
    r["success"] and close(r["result"], 0.25)
)

r = m.sample_exponential(10, 2)
check(
    "Exponential sampling",
    r["success"] and len(r["result"]) == 10
)

r = m.plot_exponential_distribution(1)
check("Exponential visualization", r["success"])


# ============================================================
# L5.8 — NORMAL
# ============================================================

r = m.normal_pdf([-1, 0, 1])
check("Normal PDF", r["success"])

r = m.normal_cdf([-1, 0, 1])
check(
    "Normal CDF",
    r["success"] and close(r["result"][1], 0.5)
)

r = m.normal_probability(-1, 1)
check(
    "Normal interval probability",
    r["success"] and close(r["result"], 0.682689492)
)

r = m.normal_mean(5)
check("Normal mean", r["success"] and r["result"] == 5)

r = m.normal_variance(2)
check("Normal variance", r["success"] and r["result"] == 4)

r = m.sample_normal(10, 5, 2)
check(
    "Normal sampling",
    r["success"] and len(r["result"]) == 10
)

r = m.plot_normal_distribution()
check("Normal visualization", r["success"])


# ============================================================
# L5.9 — T DISTRIBUTION
# ============================================================

r = m.t_pdf([-1, 0, 1], 10)
check("t PDF", r["success"])

r = m.t_cdf([-1, 0, 1], 10)
check(
    "t CDF",
    r["success"] and close(r["result"][1], 0.5)
)

r = m.t_probability(-1, 1, 10)
check("t interval probability", r["success"])

r = m.t_mean(10)
check("t mean", r["success"] and r["result"] == 0)

r = m.t_variance(10)
check(
    "t variance",
    r["success"] and close(r["result"], 1.25)
)

r = m.t_quantile(0.975, 10)
check(
    "t quantile",
    r["success"] and close(r["result"], 2.22813885, 1e-5)
)

r = m.sample_t(10, 10)
check(
    "t sampling",
    r["success"] and len(r["result"]) == 10
)

r = m.plot_t_distribution(10)
check("t visualization", r["success"])


# ============================================================
# L5.9 — CHI-SQUARE DISTRIBUTION
# ============================================================

r = m.chi_square_pdf([0, 1, 2], 5)
check("Chi-square PDF", r["success"])

r = m.chi_square_cdf([0, 1, 2], 5)
check(
    "Chi-square CDF",
    r["success"]
    and np.all(r["result"] >= 0)
    and np.all(r["result"] <= 1)
)

r = m.chi_square_probability(1, 3, 5)
check("Chi-square interval probability", r["success"])

r = m.chi_square_mean(5)
check("Chi-square mean", r["success"] and r["result"] == 5)

r = m.chi_square_variance(5)
check("Chi-square variance", r["success"] and r["result"] == 10)

r = m.chi_square_quantile(0.95, 5)
check(
    "Chi-square quantile",
    r["success"] and close(r["result"], 11.07049769, 1e-5)
)

r = m.sample_chi_square(10, 5)
check(
    "Chi-square sampling",
    r["success"] and len(r["result"]) == 10
)

r = m.plot_chi_square_distribution(5)
check("Chi-square visualization", r["success"])


# ============================================================
# L5.10 — SAMPLING
# ============================================================

population = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

r = m.random_sample(population, 5)
check(
    "Random sampling",
    r["success"] and len(r["result"]) == 5
)

r = m.sample_mean(population)
check(
    "Sample mean",
    r["success"] and r["result"] == 55
)

r = m.sample_variance(population)
check(
    "Sample variance",
    r["success"] and close(r["result"], 916.6666667)
)

r = m.standard_error(population)
check("Standard error", r["success"])

r = m.sampling_distribution_means(
    population,
    5,
    1000
)
check(
    "Sampling distribution",
    r["success"] and len(r["result"]) == 1000
)

r = m.plot_sampling_distribution(
    population,
    5
)
check("Sampling distribution visualization", r["success"])


# ============================================================
# L5.11 — CLT
# ============================================================

r = m.central_limit_theorem(
    population,
    5,
    1000
)
check(
    "Central Limit Theorem simulation",
    r["success"] and len(r["result"]) == 1000
)

r = m.clt_standard_error(
    population,
    5
)
check("CLT standard error", r["success"])

r = m.plot_clt(
    population,
    5
)
check("CLT visualization", r["success"])


# ============================================================
# L5.12 — LLN
# ============================================================

r = m.law_of_large_numbers(
    population,
    1000
)
check(
    "Law of Large Numbers",
    r["success"] and len(r["result"]) == 1000
)

r = m.plot_law_of_large_numbers(
    population,
    1000
)
check("LLN visualization", r["success"])


# ============================================================
# L5.13 — STATISTICAL TESTS
# ============================================================

r = m.one_sample_t_test(
    [12, 13, 14, 15, 16],
    10
)
check(
    "One-sample t-test",
    r["success"]
    and "statistic" in r["result"]
    and "p_value" in r["result"]
)

r = m.independent_t_test(
    [10, 11, 12, 13, 14],
    [20, 21, 22, 23, 24]
)
check(
    "Independent t-test",
    r["success"]
    and "statistic" in r["result"]
    and "p_value" in r["result"]
)

r = m.paired_t_test(
    [10, 12, 14, 16, 18],
    [12, 14, 16, 18, 20]
)
check(
    "Paired t-test",
    r["success"]
    and "statistic" in r["result"]
    and "p_value" in r["result"]
)

r = m.chi_square_goodness_of_fit(
    [20, 30, 25, 25]
)
check(
    "Chi-square goodness-of-fit",
    r["success"]
    and "statistic" in r["result"]
    and "p_value" in r["result"]
)

r = m.chi_square_independence(
    [[20, 30], [30, 20]]
)
check(
    "Chi-square independence",
    r["success"]
    and "statistic" in r["result"]
    and "p_value" in r["result"]
    and "degrees_of_freedom" in r["result"]
    and "expected" in r["result"]
)

r = m.interpret_p_value(0.03)
check(
    "p-value interpretation — reject",
    r["success"]
    and r["result"]["decision"]
    == "Reject the null hypothesis."
)

r = m.interpret_p_value(0.20)
check(
    "p-value interpretation — fail to reject",
    r["success"]
    and r["result"]["decision"]
    == "Fail to reject the null hypothesis."
)


# ============================================================
# EDGE / FAILURE TESTS
# ============================================================

r = m.calculate_mean([])
check("Empty mean rejected", not r["success"])

r = m.calculate_probability(6, 5)
check("Invalid probability rejected", not r["success"])

r = m.uniform_pdf([1], 5, 2)
check("Invalid uniform range rejected", not r["success"])

r = m.bernoulli_probability(2, 0.5)
check("Invalid Bernoulli outcome rejected", not r["success"])

r = m.binomial_probability(11, 10, 0.5)
check("Invalid binomial outcome rejected", not r["success"])

r = m.geometric_probability(0, 0.5)
check("Invalid geometric trial rejected", not r["success"])

r = m.poisson_probability(-1, 3)
check("Invalid Poisson event rejected", not r["success"])

r = m.exponential_mean(0)
check("Invalid exponential rate rejected", not r["success"])

r = m.normal_variance(0)
check("Invalid normal deviation rejected", not r["success"])

r = m.t_quantile(1.5, 10)
check("Invalid t probability rejected", not r["success"])

r = m.chi_square_mean(0)
check("Invalid chi-square df rejected", not r["success"])

r = m.paired_t_test([1, 2], [1, 2, 3])
check("Mismatched paired samples rejected", not r["success"])


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL L5 RESULT")
print("=" * 70)
print(f"PASSED : {passed}")
print(f"FAILED : {failed}")
print(f"TOTAL  : {passed + failed}")

if failed == 0:
    print()
    print("🔥🔥🔥 L5 FULL INTEGRATION TEST PASSED 🔥🔥🔥")
    print("Sensei Probability & Statistics Engine is READY.")
else:
    print()
    print("⚠️ L5 HAS FAILURES — DO NOT LOCK L5 YET.")