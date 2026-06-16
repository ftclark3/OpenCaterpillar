# same imports that we use in the openawsem functionTerms files
from openmm.app import *
from openmm import *
from openmm.unit import *

def caterpillar_contact_term(
        cat, 
        r_max = 1.2, # this is like the 'eta' that we use to calculate rho in 
        OMEGA = 21.0, # this is the critical number of neighbors for the burial potential
        #burial_kappa = 4.0 # burial_kappa not in ivan's model but needed for MD (or may be needed to achieve good efficiency); using same value as awsem
        k_contact=4.184, k_burial=4.184,
        parametersLocation=None, 
        forceGroup=22,
        gammaName="gamma.dat", burialGammaName="burial_gamma.dat"):
    


    nwell = 1
    gamma_ijm = np.zeros((nwell, 20, 20))


    # read in gamma info-- will create mock arrays for now


    m = 0  
    count = 0
    for i in range(20):
        for j in range(i, 20):
            gamma_ijm[m][i][j] = gamma_direct[count][0]
            gamma_ijm[m][j][i] = gamma_direct[count][0]
            count += 1
    



    # create Force
    contact = CustomGBForce()
    contact.addTabulatedFunction("gamma_ijm", Discrete3DFunction(nwell, 20, 20, gamma_ijm.T.flatten()))
    contact.addTabulatedFunction("burial_gamma_ij", Discrete2DFunction(20, 3, burial_gamma.T.flatten()))
    contact.addTabulatedFunction("burial_rho_ij", Discrete1DFunction(20, burial_rho))
    contact.addTabulatedFunction("inSameChain",Discrete2DFunction(oa.nres, oa.nres, np.array([[int(inSameChain(i,j,oa.chain_starts,oa.chain_ends)) for j in range(oa.nres)] for i in range(oa.nres)]).T.flatten()))
    contact.addPerParticleParameter("resName")
    contact.addPerParticleParameter("resId")
    contact.addPerParticleParameter("isSite")
    contact.addComputedValue("rho", # minimum sequence separation? 
        f"isSite1*isSite2*step(abs(resId1-resId2)-2)*0.25*(1+tanh({eta}*(r-{r_min})))*(1+tanh({eta}*({r_max}-r)))", 
        CustomGBForce.ParticlePair)

    # replace cb with ca for GLY
    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    none_cb_fixed = [i for i in range(oa.natoms) if i not in cb_fixed]
    assert len(cb_fixed) == oa.nres, f"Number of atoms in cb_fixed (non-GLY CB and GLY CA atoms), {len(cb_fixed)}, does not match number of residues {oa.nres}."
    # print(oa.natoms, len(oa.resi), oa.resi, seq)
    for i in range(oa.natoms):
        contact.addParticle([gamma_se_map_1_letter[cat.seq[cat.resi[i]]], cat.resi[i], int(i in cb_fixed),]) 


    
    # Add pairwise term
    energy_string = f"-isSite1*isSite2*{k_contact}*gamma_ijm(1,resId1,resId2)*\
        (1-1/(1+exp(25*({r_max}-1.2))))"
    contact.addEnergyTerm(energy_string, CustomGBForce.ParticlePair)

    # SET UP THE BURIAL FORCE
    contact.addEnergyTerm(
        f"isSite*{k_burial}*burial_gamma_ij(resName)*max(0,{OMEGA}-rho)",
         CustomGBForce.SingleParticle)


    # boilerplate
    contact.setCutoffDistance(1.5) # needs to be bigger than the 12 angstroms
    if cat.periodic_box:
        contact.setNonbondedMethod(contact.CutoffPeriodic)
    else:
        contact.setNonbondedMethod(contact.CutoffNonPeriodic)
    contact.setForceGroup(forceGroup)
    
    return contact


def caterpillar_hb_term(cat, e_H, nu, sigma, forceGroup):
    # create Force
    hb = CustomHbondForce(
        f"{e_H}*((cos(theta1)*cos(theta2))^{nu})*(5*(div^12)-6*(div^10))\
            ;div={sigma}/distance(a2,d2);theta1=angle(a1,a2,d2);theta2=angle(d1,d2,a2)")
    # add "donors" and "acceptors" that may interact with each other
    for resindex, oneletter in enumerate(cat.seq):
        assert cat.o[resindex] != -1 # every residue should have an O
        if resindex in cat.chain_starts: # residue shouldn't have an N or H
            assert cat.n[resindex] == -1
            assert cat.h[resindex] == -1
            assert cat.c[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1) # -1 is a placeholder for the optional 3rd particle
        elif resindex in cat.chain_ends: # residue shouldn't have a C
            assert cat.c[resindex] == -1
            assert cat.n[resindex] != -1
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1)
        else: # n and c should always be present, also h if not proline
            assert cat.c[resindex] != -1 
            assert cat.n[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1)
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1) 
    # boilerplace
    hb.setCutoffDistance(1.0) # this potential decays pretty quickly; cutoff considers d1-a1 distance
    if cat.periodic_box:
        hb.setNonbondedMethod(hb.CutoffPeriodic)
    else:
        hb.setNonbondedMethod(hb.CutoffNonPeriodic)
    hb.setForceGroup(forceGroup)                
    return hb
    
