# same imports that we use in the openawsem functionTerms files
from openmm.app import *
from openmm import *
from openmm.unit import *


# copied from openawsem hydrogenBondTerms.py
def find_chain_index(res, chain_starts, chain_ends):
    """
    Find the index of the chain that contains the residue with index `res`.
    """
    chain_index = [int(chain_start<=res and res<=chain_end) for chain_start,chain_end in zip(chain_starts,chain_ends)]
    assert sum(chain_index) == 1, f"res: {res}, chain_starts: {chain_starts}, chain_ends: {chain_ends}, list: {chain_index}"
    return chain_index.index(1)


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
    contact.addPerParticleParameter("resName")
    contact.addPerParticleParameter("resId")
    contact.addPerParticleParameter("isSite")
    contact.addPerParticleParameter("chainId")
    contact.addPerParticleParameter("e_Sol") # steepness of solvation penalty, kind of like a burial gamma
    
    # Logic that evaluates to 1 if two residues should be included in the
    # rho/omega calculation and the contact energy calculation,
    # otherwise 0. This logic enforces a minimum sequence separation of 3
    # for two residues within the same chain (but not minimum for interchain pairs)
    seqsep = 'max(step(abs(resId1-resId2)-3),step(abs(chainId2-chainId1)-1))' 
    
    # tell the Force how to calculate the density/number of neighbors,
    # called "Omega^i" in the coluzza literature and "rho" or "rho_i" in the wolynes literature
    contact.addComputedValue("rho", 
        f"isSite1*isSite2*{seqsep}*0.25*(1+tanh({eta}*(r-{r_min})))*(1+tanh({eta}*({r_max}-r)))", 
        CustomGBForce.ParticlePair)

    # we must add all Particles in the System to the Force, but
    # we activate isSite for one particle per residue 
    # replace cb with ca for GLY
    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    none_cb_fixed = [i for i in range(oa.natoms) if i not in cb_fixed]
    assert len(cb_fixed) == oa.nres, f"Number of atoms in cb_fixed (non-GLY CB and GLY CA atoms), {len(cb_fixed)}, does not match number of residues {oa.nres}."
    # print(oa.natoms, len(oa.resi), oa.resi, seq)
    for i in range(oa.natoms):
        contact.addParticle([
            gamma_se_map_1_letter[cat.seq[cat.resi[i]]], cat.resi[i], int(i in cb_fixed), 
            find_chain_index(cat.resi[i], cat.chain_starts, cat.chain_ends)
            ]) 


    
    # Add pairwise term
    energy_string = f"-isSite1*isSite2*{k_contact}*gamma_ijm(1,resId1,resId2)*\
        (1-1/(1+exp(25*({r_max}-1.2))))"
    contact.addEnergyTerm(energy_string, CustomGBForce.ParticlePair)

    # SET UP THE BURIAL FORCE
    contact.addEnergyTerm(
        f"isSite*{k_burial}*max(0,e_Sol*({OMEGA}-rho))", # this is what's written in coluzza 2011, but the code might be different
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
    # IMPLEMENTATION NOTE:
    # Currently, it is probably necessary for all instances of nonbonded forces
    # (NonbondedForce, CustomNonbondedForce, CustomGBForce) to share the same exclusion
    # list (see https://github.com/openmm/openmm/issues/5165). Even if the code compiles
    # on a particular Platform, it is possible that the results would be unreliable.
    # For this reason, it is risky to add Exclusions to nonbonded forces such as
    # caterpillar_contact_term, and we have written these terms to avoid using exclusions.
    # However, CustomHbondForce, used here, is not implemented as a nonbonded force
    # and doesn't use a neighbor list (github.com/openmm/openmm/pull/4060).
    # Unfortunately, the number of exclusions that may be added per donor/acceptor
    # is limited on some Platforms. 
    # So we will build the exclusions into the potential energy expression.
    #
    # create Force
    hb = CustomHbondForce(
        f"{e_H}*max(step(abs(resId2-resId1)-4),step(abs(chainId2-chainId1)-1))*\
            ((cos(theta1)*cos(theta2))^{nu})*(5*(div^12)-6*(div^10))\
            ;div={sigma}/distance(a2,d2);theta1=angle(a1,a2,d2);theta2=angle(d1,d2,a2)")
    hb.addPerDonorParameter('resId')
    hb.addPerAcceptorParameter('resId')
    hb.addPerDonorParameter('chainID')
    hb.addPerAcceptorParameter('chainID')
    # add "donors" and "acceptors" that may interact with each other
    for resindex, oneletter in enumerate(cat.seq):
        assert cat.o[resindex] != -1 # every residue should have an O
        if resindex in cat.chain_starts: # residue shouldn't have an N or H
            assert cat.n[resindex] == -1
            assert cat.h[resindex] == -1
            assert cat.c[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1, # -1 is a placeholder for the optional 3rd particle
                [resindex, find_chain_index(cat.chain_starts, cat.chain_ends)]) 
        elif resindex in cat.chain_ends: # residue shouldn't have a C
            assert cat.c[resindex] == -1
            assert cat.n[resindex] != -1
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1,
                    [resindex, find_chain_index(cat.chain_starts, cat.chain_ends)])
        else: # n and c should always be present, also h if not proline
            assert cat.c[resindex] != -1 
            assert cat.n[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1,
                           [resindex, find_chain_index(cat.chain_starts, cat.chain_ends)])
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1,
                            [resindex, find_chain_index(cat.chain_starts, cat.chain_ends)]) 
    # boilerplate
    hb.setCutoffDistance(1.0) # this potential decays pretty quickly; cutoff considers d1-a1 distance
    if cat.periodic_box:
        hb.setNonbondedMethod(hb.CutoffPeriodic)
    else:
        hb.setNonbondedMethod(hb.CutoffNonPeriodic)
    hb.setForceGroup(forceGroup)                
    return hb
    
