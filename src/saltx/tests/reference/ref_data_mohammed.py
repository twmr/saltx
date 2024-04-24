# Copyright (C) 2024 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""This script contains the data from the plots of the Nonlinear EP paper
https://doi.org/10.1063/5.0105963."""

import numpy as np

# Note that in fig S1 in the SM, D1 is fixed to 0.95 (there is a typo in the caption of
# the figure in the SM)

figs1_freq_mode1_data = np.array(
    [
        (0.0014735676412201415, 1.001293299982386),
        (0.013234040815910009, 1.0012941723425597),
        (0.02499451399059993, 1.0012941723425597),
        (0.036754987165289854, 1.0012941723425597),
        (0.04851546033997978, 1.0012941723425597),
        (0.0602759335146697, 1.0012941723425597),
        (0.07203640668935962, 1.0012941723425597),
        (0.08379687986404954, 1.0012941723425597),
        (0.09555735303873947, 1.0012941723425597),
        (0.10731782621342939, 1.0012941723425597),
        (0.11907829938811926, 1.0012941723425597),
        (0.13083877256280918, 1.0013011512239482),
        (0.1425992457374991, 1.001303768304469),
        (0.15435971891218903, 1.001303768304469),
        (0.16612019208687895, 1.001303768304469),
        (0.17788066526156887, 1.001303768304469),
        (0.1896411384362588, 1.001303768304469),
        (0.20140161161094872, 1.001303768304469),
        (0.21316208478563864, 1.001303768304469),
        (0.22492255796032856, 1.0013063853849897),
        (0.23668303113501843, 1.0013133642663783),
        (0.2484435043097084, 1.0013133642663783),
        (0.2602039774843983, 1.0013133642663783),
        (0.27196445065908825, 1.0013133642663783),
        (0.2837249238337781, 1.0013133642663783),
        (0.295485397008468, 1.0013133642663783),
        (0.30724587018315797, 1.0013133642663783),
        (0.31900634335784783, 1.0013133642663783),
        (0.3307668165325378, 1.0013133642663783),
        (0.3425272897072277, 1.0013133642663783),
        (0.35428776288191766, 1.0013133642663783),
        (0.3660482360566075, 1.0013133642663783),
        (0.3778087092312975, 1.0013133642663783),
        (0.38956918240598737, 1.0013133642663783),
        (0.40132965558067724, 1.0013133642663783),
        (0.4130901287553672, 1.0013133642663783),
        (0.4248506019300571, 1.0013133642663783),
        (0.43661107510474706, 1.0013133642663783),
        (0.4483715482794369, 1.0013133642663783),
        (0.4601320214541269, 1.0013133642663783),
        (0.47189249462881677, 1.0013133642663783),
        (0.48365296780350675, 1.0013133642663783),
        (0.4954134409781966, 1.0013133642663783),
        (0.5071739141528865, 1.0013133642663783),
        (0.5189343873275765, 1.0013133642663783),
        (0.5306948605022663, 1.0013133642663783),
        (0.5424553336769563, 1.0013133642663783),
        (0.5542158068516462, 1.0013124919062046),
        (0.5634370869545281, 1.0012258011139559),
        (0.5639907455942456, 1.001003551781878),
        (0.5649071461013643, 1.0008047782851854),
        (0.5652126129370706, 1.0005689917925569),
        (0.567198147369161, 1.000331834448227),
        (0.5681145478762798, 1.0000826821515107),
        (0.5702528157262233, 0.9998571770466422),
        (0.5734602175011388, 0.999607415482503),
        (0.5765339775354328, 0.999353389046404),
        (0.5800694090373281, 0.9990984417856774),
        (0.5852206906758287, 0.9988340076080632),
        (0.589497226375716, 0.9985977070460468),
        (0.5937737620756032, 0.9983913938649969),
        (0.5985848647379763, 0.9981735655296557),
        (0.6039305343628354, 0.9979595755790784),
        (0.6098107709501803, 0.9977404677821494),
        (0.6161283805068319, 0.9975325552741146),
        (0.6253132128622716, 0.9972859390530456),
        (0.6349354181870179, 0.9970371637405471),
        (0.6440230565492783, 0.9968096594769474),
        (0.6536452618740246, 0.9965900185710235),
        (0.6638020341612568, 0.9963762418640442),
        (0.6750279403734608, 0.9961681839626471),
        (0.6867884135481507, 0.995957945160816),
        (0.6985488867228407, 0.9957677706429772),
        (0.7103093598975305, 0.9955924262480891),
        (0.7220698330722205, 0.995433656696499),
        (0.7338303062469104, 0.9952836107466444),
        (0.7455907794216002, 0.9951483949197407),
        (0.7573512525962903, 0.995019285614052),
        (0.7691117257709802, 0.9949015169906196),
        (0.7808721989456701, 0.9947924719689231),
        (0.7926326721203599, 0.9946921505489622),
        (0.8043931452950498, 0.9945996803705635),
        (0.8161536184697399, 0.9945141890735534),
        (0.8279140916444298, 0.9944339319375847),
        (0.8396745648191196, 0.9943615260431782),
        (0.8514350379938095, 0.9942952266699867),
        (0.8631955111684996, 0.9942341614578366),
        (0.8749559843431894, 0.9941809474872486),
        (0.8867164575178793, 0.994131222957355),
        (0.8984769306925692, 0.9940876049486764),
        (0.910237403867259, 0.9940466040205185),
        (0.9219978770419491, 0.9940143266940963),
        (0.933758350216639, 0.9939872835287156),
        (0.9455188233913289, 0.9939619850836819),
        (0.9572792965660187, 0.9939419207996898),
        (0.9690397697407088, 0.9939314524776068),
        (0.9808002429153987, 0.9939244735962183),
        (0.9925607160900886, 0.993920984155524),
        (0.9995100866024053, 0.993914877634309),
    ]
)

figs1_freq_mode2_data = np.asarray(
    [
        (0.5642092746730116, 1.0012113127001065),
        (0.5640755974930528, 1.0014732969648665),
        (0.5650598795192174, 1.001813725137363),
        (0.5661544690138314, 1.0021194250667593),
        (0.5688273038262609, 1.0024339593737863),
        (0.5714746830690483, 1.0027349374806558),
        (0.5745293514261106, 1.0030166475052789),
        (0.577736753201026, 1.003245351264117),
        (0.5814787219384274, 1.0034892486293117),
        (0.5857552576383146, 1.003730347172283),
        (0.5904475676423576, 1.003970379497264),
        (0.5953774629630609, 1.0042065467820318),
        (0.6007994992968465, 1.004455356365823),
        (0.6071379361377508, 1.0047135334362398),
        (0.6135527396875817, 1.0049406378680932),
        (0.6199675432374125, 1.0051453517221582),
        (0.6269169137497294, 1.0053546350818943),
        (0.6349354181870179, 1.0055763703445841),
        (0.6434884895867924, 1.005793478982782),
        (0.6520415609865668, 1.0059913956971613),
        (0.6616637663113132, 1.0061996280705932),
        (0.6723551055610313, 1.0064145776173614),
        (0.6835810117732353, 1.0066199312022204),
        (0.6953414849479252, 1.0068170846014477),
        (0.7071019581226151, 1.0070011525980715),
        (0.7188624312973051, 1.0071669010310504),
        (0.730622904471995, 1.0073230535021198),
        (0.7423833776466849, 1.007466120570586),
        (0.7541438508213748, 1.0076013363974896),
        (0.7659043239960647, 1.007725211542137),
        (0.7776647971707545, 1.0078386183647012),
        (0.7894252703454446, 1.0079441739457036),
        (0.8011857435201345, 1.0080410059249703),
        (0.8129462166948244, 1.0081308590228482),
        (0.8247066898695142, 1.008212860879164),
        (0.8364671630442043, 1.008290500934612),
        (0.8482276362188942, 1.0083594173883241),
        (0.859988109393584, 1.0084248444013422),
        (0.8717485825682739, 1.008482420172798),
        (0.8835090557429638, 1.0085338894230387),
        (0.8952695289176539, 1.0085792521520645),
        (0.9070300020923437, 1.0086202530802224),
        (0.9187904752670336, 1.0086542751269918),
        (0.9305509484417235, 1.0086874248135875),
        (0.9423114216164136, 1.0087092338179269),
        (0.9540718947911034, 1.008735404623134),
        (0.9658323679657933, 1.0087450005850433),
        (0.9775928411404832, 1.0087545965469527),
        (0.989353314315173, 1.008759830707994),
        (0.9979063857149476, 1.0087636690927577),
    ]
)

figs1_intensity_single_mode_data = """
0.003691836734693865, 0.17495441928750033,
0.018702721088435376, 0.17480716628816556,
0.03371360544217686, 0.17407090129149178,
0.04872448979591837, 0.1737763952928223,
0.06373537414965985, 0.17318738329548322,
0.07874625850340133, 0.1727088110476453,
0.09375714285714282, 0.17219342554997363,
0.10876802721088435, 0.17156760030280094,
0.12377891156462584, 0.17086814855596083,
0.13878979591836732, 0.17024232330878813,
0.1538006802721088, 0.16954287156194803,
0.16881156462585034, 0.16873298006560689,
0.18382244897959182, 0.1680703415686005,
0.1988333333333333, 0.1671868235725919,
0.2138442176870748, 0.16637693207625076,
0.22885510204081633, 0.16538297433074123,
0.2438659863945578, 0.16438901658523158,
0.2588768707482993, 0.1632846190902209,
0.2738877551020408, 0.16229066134471132,
0.28889863945578226, 0.1610021976005322,
0.3039095238095238, 0.15971373385635312,
0.3189204081632652, 0.15824120386300555,
0.33393129251700676, 0.15680548711949166,
0.3489421768707483, 0.15525933062647673,
0.3639530612244897, 0.15352910788429341,
0.37896394557823126, 0.15157800564310786,
0.3939748299319728, 0.1494796504025876,
0.40898571428571423, 0.14712360241323152,
0.42399659863945577, 0.1445098616750396,
0.4390074829931972, 0.14156480168834445,
0.45401836734693873, 0.13825160920331248,
0.46902925170068027, 0.13434940472094148,
0.4840401360544217, 0.12982137499139773,
0.49905102040816324, 0.12433620076617802,
0.5140619047619046, 0.11748893629711199,
0.527708163265306, 0.10961090083270253,
0.5386251700680271, 0.10128058829890774,
0.547495238095238, 0.09240070653545296,
0.5543183673469387, 0.08318819076457235,
0.5593674829931974, 0.07413765329295996,
0.5656902494331066, 0.06618046934140809,
0.5781993197278912, 0.07374427742330847,
0.5877517006802722, 0.08276878266825283,
0.5973040816326529, 0.09208253487617613,
0.6068564625850339, 0.10133843769150364,
0.6164088435374149, 0.11082573807721424,
0.6252789115646258, 0.11973454453696697,
0.6334666666666666, 0.1280359323744638,
0.6416544217687075, 0.13647230212801753,
0.6505244897959184, 0.14553537363469232,
0.6600768707482993, 0.1552540715907862,
0.66894693877551, 0.1644810497098158,
0.6771346938775509, 0.17291741946336953,
0.6860047619047618, 0.18221188854042758,
0.6948748299319727, 0.19140994196315925,
0.7030625850340135, 0.19998129363276979,
0.7112503401360544, 0.20862013626040882,
0.7194380952380952, 0.2172589788880478,
0.727625850340136, 0.22583033055765833,
0.7358136054421769, 0.23460415510135418,
0.7440013605442175, 0.2433104886870216,
0.7521891156462583, 0.25194933131466063,
0.7603768707482992, 0.2607906468163849,
0.76856462585034, 0.26949698040205233,
0.7767523809523809, 0.2781358230296913,
0.7849401360544217, 0.2871121204474725,
0.7931278911564625, 0.29588594499116827,
0.8013156462585034, 0.3045922785768357,
0.8095034013605442, 0.31363606695264534,
0.8176911564625851, 0.32240989149634114,
0.8258789115646259, 0.3313861889141223,
0.8340666666666667, 0.3400250315417613,
0.8422544217687074, 0.3490013289595424,
0.8504421768707482, 0.3577751535032383,
0.858629931972789, 0.36668395996299097,
0.8668176870748299, 0.37566025738077213,
0.8750054421768707, 0.3845015728824964,
0.8831931972789115, 0.39347787030027753,
0.8913809523809524, 0.4022516948439734,
0.8995687074829932, 0.41122799226175455,
0.9077564625850341, 0.4200018168054504,
0.9159442176870749, 0.4291130961392884,
0.9241319727891155, 0.43788692068298424,
0.9323197278911564, 0.4467282361847085,
0.9405074829931972, 0.45563704264446125,
0.948695238095238, 0.4645458491042139,
0.9568829931972789, 0.4733196736479098,
0.9650707482993197, 0.48216098914963407,
0.9732585034013606, 0.49120477752544367,
0.9814462585034014, 0.4999786020691395,
0.9896340136054422, 0.5088874085288922,
0.9971394557823128, 0.5163384102952309
"""
figs1_intensity_single_mode_data = np.fromstring(
    figs1_intensity_single_mode_data, sep=","
).reshape((-1, 2))


figs1_intensity_second_mode_data = """
0.5641693811074919, 0.062452107279693636,
0.5720168166684676, 0.07279350495017534,
0.5829219218755628, 0.08472981343276043,
0.5938270270826578, 0.09666612191534563,
0.604732132289753, 0.1088696611848543,
0.6156372374968482, 0.12116227738333746,
0.6265423427039432, 0.13354397051079508,
0.6374474479110384, 0.1458365867092783,
0.6483525531181333, 0.15759474133391438,
0.6592576583252285, 0.17015458831932107,
0.6701627635323237, 0.18226905065985521,
0.6810678687394187, 0.19456166685833837,
0.6919729739465139, 0.20676520612784705,
0.7028780791536091, 0.21887966846838114,
0.7137831843607041, 0.2313504385248134,
0.7246882895677993, 0.24337582393637303,
0.7355933947748943, 0.2556684401348562,
0.7464984999819895, 0.2679610563333394,
0.7574036051890847, 0.28025367253182254,
0.7683087103961797, 0.29227905794338216,
0.7792138156032749, 0.30466075107083984,
0.7901189208103698, 0.3168642903403485,
0.801024026017465, 0.3290678296098572,
0.81192913122456, 0.34136044580834035,
0.8228342364316552, 0.35356398507784903,
0.8337393416387504, 0.3660347551342812,
0.8446444468458454, 0.3777929097589173,
0.8555495520529406, 0.39044183367329854,
0.8664546572600356, 0.40246721908485816,
0.8773597624671308, 0.4147598352833413,
0.888264867674226, 0.42723060533977353,
0.899169972881321, 0.4395232215382567,
0.9100750780884161, 0.45154860694981636,
0.9209801832955111, 0.463573992361376,
0.9318852885026063, 0.4756884547019101,
0.9427903937097015, 0.48789199397141875,
0.9536954989167965, 0.49973922552502936,
0.9646006041238917, 0.5120318417235125,
0.9755057093309867, 0.5241463040640466,
0.9864108145380819, 0.5359935356176573,
0.996407160977919, 0.5467896594093685,
0.9993485342019544, 0.550191570881226
"""
figs1_intensity_second_mode_data = np.fromstring(
    figs1_intensity_second_mode_data, sep=","
).reshape((-1, 2))


def main():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(
        figs1_freq_mode2_data[:, 0],
        figs1_freq_mode2_data[:, 1],
        "gx",
        alpha=0.9,
        label="mode2",
    )
    ax.plot(
        figs1_freq_mode1_data[:, 0],
        figs1_freq_mode1_data[:, 1],
        "rx",
        alpha=0.9,
        label="mode1",
    )

    ax.axvline(x=0.566, c="k")
    ax.text(
        0.566,
        0.01,
        "EP: D2/D1=0.566",
        color="k",
        ha="right",
        va="bottom",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )
    ax.grid(True)
    ax.set_xlabel("D2/D1")
    ax.set_ylabel("fa/c")
    fig.legend()

    fig, ax = plt.subplots()
    data = figs1_intensity_single_mode_data
    ax.plot(data[:, 0], data[:, 1], "rx", label="mode1")

    data2 = figs1_intensity_second_mode_data
    ax.plot(data2[:, 0], data2[:, 1], "gx", label="mode2")

    ax.axvline(x=0.566, c="k", linewidth=0.2)
    ax.text(
        0.566,
        0.99,
        "EP: D2/D1=0.566",
        color="k",
        ha="right",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )

    ax.grid(True)
    ax.set_xlabel("D2/D1")
    ax.set_ylabel("Pout")
    fig.legend()

    plt.show()


if __name__ == "__main__":
    main()
