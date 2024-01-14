#!/usr/bin/env python3
#
# Shows how to add constraints to a fitting task, using
# the Slack model as an example.
#

from teflow.kappa import Variable, KappaSlack


# data to be fitted
Temper = [300,    400,    500,    600,    700,    800,   ]
Kappa =  [5.0460, 3.6354, 3.1787, 2.3826, 2.0858, 1.8614,]

# Parameters required by Slack model. Here, the prepositive coefficient
# depends on the Gruneisen parameter of the material.
DebyeTemper = 190       # in Kelvin
AtomicWeight = 150      # in atomic mass units (amu, or g/mol)
AtomicVolume = 30       # in cubic angstroms (A^3)
Gruneisen = Variable(
    tag='gm',
    initial=2,
)
Coefficient = Variable(
    tag='A',
    scale=1E-6,
    constraint=lambda x: 2.43/(1-0.514/x+0.228/x/x),
    depends=(Gruneisen,),
)

# Initialize the model
model = KappaSlack(td=DebyeTemper,
                   gm=Gruneisen,
                   Ma=AtomicWeight,
                   Va=AtomicVolume,
                   A=Coefficient,)

# Run fitting
model.fit(Temper, Kappa, variables=(Gruneisen,))
print('Fitting Results:')
print(f' Gruneisen parameter: {Gruneisen.value:.4f}')
print(f' Prepositive coefficient: {Coefficient.value*Coefficient.scale:.4E}')

# Predict
Kappa_pred = model(Temper)
print(' Temperatures, Kappa_Exp, Kappa_Pred, and Difference:')
for T, KE, KP, DF in zip(Temper, Kappa, Kappa_pred, Kappa_pred-Kappa):
    print(f'     {T:.2f}    {KE:.4f}    {KP:.4f}    {DF:8.4f}')

# Predict kappa when Gruneisen parameter is equal to 2
# Hint: When we update 'Gruneisen' manually, downstream 'Coefficient'
#       will be updated automatically.
Gruneisen.value = 2
print('\nPredicted kappa when Gruneisen parameter is 2:')
print(f' Gruneisen parameter: {Gruneisen.value:.4f}')
print(f' Prepositive coefficient: {Coefficient.value*Coefficient.scale:.4E}')
Kappa_pred_2 = model(Temper)
print(' Temperatures & Kappa_pred_2:')
for T, KP2 in zip(Temper, Kappa_pred_2):
    print(f'     {T:.2f}    {KP2:.4f}')
