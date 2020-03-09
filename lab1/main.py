from numpy import mean, std
from scipy.stats import chi2_contingency, norm, uniform, gamma, expon
import matplotlib.pyplot as plt

template = list()
k = 10
statUniform, statGamma, statNormal, statExp  = [None] * 4

with open('exponential.csv', 'r') as file:
    for line in file.readlines():
        template.append(float(line))

# with open('gamma.csv', 'r') as file:
#     for line in file.readlines():
#         template.append(float(line))
# with open('normal.csv', 'r') as file:
#     for line in file.readlines():
#         template.append(float(line))

# with open('uniform.csv', 'r') as file:
#     for line in file.readlines():
#         template.append(float(line))

def calculateFrequencesInCategories(arr, k):
    h = (max(arr) - min(arr)) / k
    deltas = [min(arr) + h * i for i in range(k)]
    rest = [sum(map(lambda x: delta <= x < delta + h, arr)) for delta in deltas]
    return rest

while not (statNormal and statExp and statUniform and statGamma):
    try:
        normal_template = norm.rvs(loc=mean(template), scale=std(template), size=len(template))
        statNormal, p_normal, _, _ = chi2_contingency([calculateFrequencesInCategories(template, k),
                                                        calculateFrequencesInCategories(normal_template, k)])

        exponential_template = expon.rvs(loc=1/mean(template), scale=mean(template), size=len(template))
        statExp, p_exp, _, _ = chi2_contingency([calculateFrequencesInCategories(template, k),
                                                  calculateFrequencesInCategories(exponential_template, k)])

        uniform_template = uniform.rvs(min(template), max(template), size=len(template))
        statUniform, p_uniform, _, _ = chi2_contingency([calculateFrequencesInCategories(template, k),
                                                          calculateFrequencesInCategories(uniform_template, k)])

        a = mean(template) ** 2 / std(template) ** 2
        b = mean(template) / std(template) ** 2
        gamma_template = gamma.rvs(a=a, scale=1 / b, size=len(template))
        statGamma, p_gamma, _, _ = chi2_contingency([calculateFrequencesInCategories(template, k),
                                                      calculateFrequencesInCategories(gamma_template, k)])
    except ValueError:
        pass

print(f'Uniform:       value = {p_uniform}   | test = {statUniform}')
print(f'Gamma:         value = {p_gamma}     | test = {statGamma}')
print(f'Normal :       value = {p_normal}    | test = {statNormal}')
print(f'Exponential:   value = {p_exp}       | test = {statExp}')

if min([statNormal, statGamma, statExp, statUniform]) == statNormal:
    print('Normal distribution')
    pdf = norm.pdf(range(int(min(template)), int(max(template))), loc=mean(template), scale=std(template))
    cdf = norm.cdf(range(int(min(template)), int(max(template))), loc=mean(template), scale=std(template))
    qdf = [1 - x for x in cdf]
elif min([statNormal, statGamma, statExp, statUniform]) == statExp:
    print('Exponential distribution')
    pdf = expon.pdf(range(int(min(template)), int(max(template))), loc=1/mean(template), scale=mean(template))
    cdf = expon.cdf(range(int(min(template)), int(max(template))), loc=1/mean(template), scale=mean(template))
    qdf = [1 - x for x in cdf]
elif min([statNormal, statGamma, statExp, statUniform]) == statUniform:
    print('Uniform distribution')
    pdf = uniform.pdf(range(int(min(template)), int(max(template))), min(template), max(template))
    cdf = uniform.cdf(range(int(min(template)), int(max(template))), min(template), max(template))
    qdf = [1 - x for x in cdf]
else:
    print('Gamma distribution')
    a = mean(template) ** 2 / std(template) ** 2
    b = mean(template) / std(template) ** 2
    pdf = gamma.pdf(range(int(min(template)), int(max(template))), a=a, scale=1/b)
    cdf = gamma.cdf(range(int(min(template)), int(max(template))), a=a, scale=1/b)
    qdf = [1 - x for x in cdf]


h = (max(template) - min(template)) / k
deltas = [min(template) + h * i for i in range(k)]
probabilities_q = [round(sum(map(lambda x: x <= delta + h, template)) / len(template), 4) for delta in deltas]
probabilities_p = [1 - p for p in probabilities_q]


plt.bar(deltas, probabilities_q, width=h, align='edge', label='Q*(t)', alpha=.2)
plt.plot(range(int(min(template)), int(max(template))), cdf, label='Q(t)')
plt.title('Q*(t) and Q(t)')
plt.legend()
plt.show()


plt.bar(deltas, probabilities_p, width=h, align='edge', label='P*(t)', alpha=.2)
plt.plot(range(int(min(template)), int(max(template))), qdf, label='P(t)')
plt.title('P*(t) and P(t)')
plt.legend()
plt.show()

plt.hist(template, density=True, label='f*(t)', alpha=.2)
plt.plot(range(int(min(template)), int(max(template))), pdf, label='f(t)')
plt.title('f*(t) and f(t)')
plt.legend()
plt.show()





lambdas = [pdf[i] / cdf[i] for i in range(len(pdf))]

plt.plot(range(int(min(template)), int(max(template))), lambdas, label='Lambda(t)')
plt.title('Lambda(t)')
plt.legend()
plt.show()

