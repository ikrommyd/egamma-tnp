from egamma_tnp import ElectronTagNProbeFromNanoAOD, ElectronTagNProbeFromNTuples


class ElePt_WPTight_Gsf:
    """Tag and Probe efficiency for HLT_ElePt_WPTight_Gsf trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 30.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 27.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        tags_pt_cut=30,
        probes_pt_cut=27,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}WPTightGsf",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}WPTightGsf",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=1,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt}_WPTight_Gsf",
            )

        return instance


class ElePt_CaloIdVT_GsfTrkIdT:
    """Tag and Probe efficiency for HLT_ElePt_CaloIdVT_GsfTrkIdT trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 30.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 112.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        tags_pt_cut=30,
        probes_pt_cut=112,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}CaloIdVTGsfTrkIdTGsf",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}CaloIdVTGsfTrkIdTGsf",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=11,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt}_CaloIdVT_GsfTrkIdT",
            )

        return instance


class ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL_Leg1:
    """Tag and Probe efficiency for the high-Pt leg of the HLT_ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt1 : int or float
            The Pt threshold of the high-Pt leg of the trigger.
        trigger_pt2 : int or float
            The Pt threshold of the low-Pt leg of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 30.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 20.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt1,
        trigger_pt2,
        *,
        from_ntuples=False,
        tags_pt_cut=30,
        probes_pt_cut=20,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt1}Ele{trigger_pt2}CaloIdLTrackIdLIsoVLLeg1L1match",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt1}Ele{trigger_pt2}CaloIdLTrackIdLIsoVLLeg1L1match",
                for_trigger=True,
                trigger_pt=trigger_pt1,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=4,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt1}_Ele{trigger_pt2}_CaloIdL_TrackIdL_IsoVL",
            )

        return instance


class ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL_Leg2:
    """Tag and Probe efficiency for the low-Pt leg of the HLT_ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt1 : int or float
            The Pt threshold of the high-Pt leg of the trigger.
        trigger_pt2 : int or float
            The Pt threshold of the low-Pt leg of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 30.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 9.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt1,
        trigger_pt2,
        *,
        from_ntuples=False,
        tags_pt_cut=30,
        probes_pt_cut=9,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt1}Ele{trigger_pt2}CaloIdLTrackIdLIsoVLLeg2",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltEle{trigger_pt1}Ele{trigger_pt2}CaloIdLTrackIdLIsoVLLeg2",
                for_trigger=True,
                trigger_pt=trigger_pt2,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=5,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt1}_Ele{trigger_pt2}_CaloIdL_TrackIdL_IsoVL",
            )

        return instance


class DoubleElePt_CaloIdL_MW_SeededLeg:
    """Tag and Probe efficiency for the seeded leg of the HLT_DoubleElePt_CaloIdL_MW trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 35.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 30.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        tags_pt_cut=35,
        probes_pt_cut=30,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltDoubleEle{trigger_pt}CaloIdLMWSeedLegL1match",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltDoubleEle{trigger_pt}CaloIdLMWSeedLegL1match",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=15,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"DoubleEle{trigger_pt}_CaloIdL_MW",
            )

        return instance


class DoubleElePt_CaloIdL_MW_UnseededLeg:
    """Tag and Probe efficiency for the unseeded leg of the HLT_DoubleElePt_CaloIdL_MW trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 35.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is 30.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        tags_pt_cut=35,
        probes_pt_cut=30,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            instance = ElectronTagNProbeFromNTuples(
                fileset=fileset,
                filter=f"passHltDoubleEle{trigger_pt}CaloIdLMWUnsLeg",
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            )
        else:
            instance = ElectronTagNProbeFromNanoAOD(
                fileset=fileset,
                filter=f"passHltDoubleEle{trigger_pt}CaloIdLMWUnsLeg",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=tags_pt_cut,
                probes_pt_cut=probes_pt_cut,
                tags_abseta_cut=2.5,
                filterbit=16,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=use_sc_eta,
                use_sc_phi=use_sc_phi,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"DoubleEle{trigger_pt}_CaloIdL_MW",
            )

        return instance
