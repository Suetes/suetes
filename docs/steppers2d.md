# Integration and Advection

The `steppers` module defines the core time integration cycle of the model, utilizing a Semi-Implicit Semi-Lagrangian (SISL) formulation. 

This approach traces the geometric trajectory of fluid parcels backward in time to solve the advection terms unconditionally stably, and treats fast-propagating sound and gravity waves implicitly via a preconditioned GMRES solver.

::: suetes.slice2d.steppers 