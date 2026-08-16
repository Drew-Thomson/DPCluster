WIP: This is an implementation of the Density Peak Clustering algorithm, as applied to Ramachandran angles.

The basic workflow is

1) Extract Ramachandran angles from MD trajectory (or other source)
2) Generate the sin and cos values for each angle
3) Run a principle component analysis for the transformed angles
4) Cluster based on the top three principle components
