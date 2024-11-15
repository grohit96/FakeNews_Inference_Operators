import numpy as np

# Path to the incorrect test set
incorrect_test_set_path = "/scratch/rganesh5/PaperGraph_release_connected/test_set_1.npy"

# Load the incorrect test set
incorrect_test_set = np.load(incorrect_test_set_path, allow_pickle=True)

# Extract all domain names from the string
domain_list = """
dailykos.com bluenationreview.com commondreams.org americablog.com dallasvoice.com desmogblog.com au.org gizmodo.com leftoverrights.com juancole.com mediaite.com leftvoice.org liberationnews.org lavendermagazine.com littlegreenfootballs.com motherjones.com bradford-delong.com newpol.org newsweek.com front.moveon.org mashable.com projectcensored.org rewire.news peacock-panache.com pfaw.org redpepper.org.uk sourcewatch.org shareblue.com americanindependent.com rantt.com rightwingwatch.org nationofchange.org worldsocialism.org nationalmemo.com thesternfacts.com thefrisky.com progressive.org themilitant.com worldcantwait.net thedailybanter.com thenation.com theroot.com vox.com wonkette.com bnonews.com eng.majalla.com alreporter.com news.abs-cbn.com bangkokpost.com altoday.com abqjournal.com belfercenter.org cgdev.org constitutioncenter.org business2community.com bettergov.org apnews.com securingdemocracy.gmfus.org ced.org afr.com ejinsight.com cambridge.org makeuseof.com military.com knowyourmeme.com maplight.org govexec.com euronews.com financialexpress.com ledevoir.com healthcarefinancenews.com newbernsj.com nber.org firstdraftnews.org factcheck.org guampdn.com merionwest.com rferl.org patheos.com nextavenue.org northkoreatimes.com russialies.com pogo.org relevantmagazine.com reporterslab.org heraldtribune.com scmp.com stratfor.com thebureauinvestigates.com taskandpurpose.com jpost.com worldpress.org wikipedia.org mywebtimes.com thejournal.ie mcall.com gao.gov usafacts.org thewrap.com truthorfiction.com aclj.org arizonadailyindependent.com americanconsequences.com 2ndvote.com christianpost.com ca-political.com conservativetoday.com colddeadhands.us acculturated.com awm.com altnewsmedia.net erlc.com dcwhispers.com freewestmedia.com ifstudies.org dailysabah.com faithwire.com knoxreport.com farleftwatch.com gatestoneinstitute.org freedomworks.org humanevents.com freerepublic.com kansaspolicy.org heritage.org oann.com nationalcenter.org renewedright.com lifenews.com opslens.com patriotnewsdaily.com thedailydefender.com savejersey.com therebel.media washingtonexaminer.com unwatch.org theresurgent.com thepostemail.com trtworld.com westernfreepress.com 10news.one christianaction.org 100percentfedup.com americanpatriotdaily.com americantoday.news ageofshitlords.com conservapedia.com allnewspipeline.com discoverthenetworks.org altright.com hoggwatch.com freakoutnation.com eaglerising.com hangthebankers.com newswars.com o4anews.com newnation.org freedomsfinalstand.com patriotfires.com takimag.com rightalerts.com dcclothesline.com speisa.com puppetstringnews.com digifection.com theduran.com theamericanmirror.com fairus.org truthuncensored.net mynewsguru.com unclesamsmisguidedchildren.com thepublicdiscourse.com
""".split()

# Create a mapping from domain name to numerical index
string_to_index_map = {domain: idx for idx, domain in enumerate(domain_list)}

# Convert strings to indices using the mapping
corrected_test_set = np.array([string_to_index_map[item] for item in incorrect_test_set if item in string_to_index_map])

# Save the corrected test set
corrected_test_set_path = "/scratch/rganesh5/PaperGraph_release_connected/test_set_1_corrected.npy"
np.save(corrected_test_set_path, corrected_test_set)

print(f"Corrected test set saved to {corrected_test_set_path}")
