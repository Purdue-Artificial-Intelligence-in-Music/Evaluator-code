import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  Modal,
  ScrollView,
  TouchableOpacity,
  Image,
  StyleSheet,
  Dimensions,
  SafeAreaView,
} from 'react-native';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Types ────────────────────────────────────────────────────────────────────

interface PostureItem {
  imageSource: any; // require(…) or { uri: '…' }
}

interface PostureTab {
  key: string;
  title: string;
  sectionTitle: string;
  description: string[];
  items?: PostureItem[];
}

// ─── Tab data ─────────────────────────────────────────────────────────────────
// Replace imageSource values with your real require(…) calls or { uri } objects.

const TABS: PostureTab[] = [
  {
    key: 'overall',
    title: 'Overall',
    sectionTitle: '1. Overall',
    description: [
      'Your shoulders should be balanced and relaxed. Tension in the body directly affects posture and sound. When the shoulders rise, the elbow, wrist, and hand shape are disrupted.',
      'Maintaining a relaxed, tension-free posture is essential for good playing.',
    ],
    items: [
      {
        imageSource: require('../assets/postures/1.Overall_example.jpg'), // ← replace
      },
    ],
  },
  {
    key: 'bowContactPoint',
    title: 'Bow Contact Point',
    sectionTitle: '2. Bow Contact Point',
    description: [
      'The bow should be placed between the fingerboard and the bridge.',
      'Especially for beginners, it is important to check that the bow is not too close to the fingerboard (too high) or too close to the bridge (too low).',
      'Advanced players may intentionally place the bow higher or lower to change tone for musical purposes.',
    ],
    items: [
      {
        imageSource: require('../assets/postures/2.Bow Contact Point examples.jpg'), // ← replace
      },
    ],
  },
  {
    key: 'bowAngle',
    title: 'Bow Angle',
    sectionTitle: '3. Bow Angle',
    description: [
      'The bow should remain perpendicular to the strings and parallel to the bridge.',
      'Each string (A, D, G, and C) has its own correct bow angle, and this angle must adjust as you move from one string to another.',
    ],
    items: [
      {
        imageSource: require('../assets/postures/3.bow angle examples.jpg'), // ← replace
      },
    ],
  },
  {
    key: 'bowHand',
    title: 'Bow Hand',
    sectionTitle: '4. Bow Hand Position',
    description: [
      'The standard bow hand posture may vary slightly depending on hand size, but the bow should be held with a slight tilt toward the left hand side. This tilt of the hand wrist and elbow is called pronation. Especially at the tip of the bow, both the hand and wrist should remain pronated.',
      'If the hand and wrist tilt to the right, this is called supination, which leads to poor tone and reduced control.',
    ],
    items: [
      {
        imageSource: require('../assets/postures/4. Bow han example.jpg'), // ← replace
      },
    ],
  },
  {
    key: 'elbow',
    title: 'Elbow',
    sectionTitle: '5. Elbow',
    description: [
      'At the frog, the elbow should be comfortably lowered and positioned close to the cello body.',
    ],
    items: [
      {
        imageSource: require('../assets/postures/5. elbow example.jpg'), // ← replace
      },
    ],
  },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

interface PostureImageCardProps {
  item: PostureItem;
}

const PostureImageCard: React.FC<PostureImageCardProps> = ({ item }) => {
  const [imageHeight, setImageHeight] = useState(200);

  useEffect(() => {
    if (item.imageSource) {
      const resolvedSource = Image.resolveAssetSource(item.imageSource);
      if (resolvedSource?.width && resolvedSource?.height) {
        const screenWidth = SCREEN_WIDTH - 40; // account for paddingHorizontal: 20
        const ratio = resolvedSource.height / resolvedSource.width;
        setImageHeight(screenWidth * ratio);
      }
    }
  }, [item.imageSource]);

  return (
    <View style={styles.imageCard}>
      <Image
        source={item.imageSource}
        style={[styles.postureImage, { height: imageHeight }]}
        resizeMode="contain"
      />
    </View>
  );
};

// ─── Main modal ───────────────────────────────────────────────────────────────

interface LearnPosturesModalProps {
  visible: boolean;
  onClose: () => void;
  initialTab?: string;
}

const LearnPosture: React.FC<LearnPosturesModalProps> = ({
  visible,
  onClose,
  initialTab = 'overall',
}) => {
  const [activeTab, setActiveTab] = useState(initialTab);
  const tabScrollRef = useRef<ScrollView>(null);

  const currentTab = TABS.find(t => t.key === activeTab) ?? TABS[0];

  return (
    <Modal visible={visible} animationType="slide" transparent={false} onRequestClose={onClose}>
      <SafeAreaView style={styles.safeArea}>
        {/* ── Header ── */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Learn postures</Text>
        </View>

        {/* ── Tab bar ── */}
        <View style={styles.tabBarWrapper}>
           <ScrollView
           horizontal
           showsHorizontalScrollIndicator={false}
           contentContainerStyle={styles.tabRow}
           >
            {TABS.map(tab => {
              const isActive = tab.key === activeTab;
              return (
                <TouchableOpacity
                  key={tab.key}
                  style={[styles.tabItem, isActive && styles.tabItemActive]}
                  onPress={() => setActiveTab(tab.key)}
                  activeOpacity={0.7}
                >
                  <Text style={[styles.tabText, isActive && styles.tabTextActive]}>
                    {tab.title}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
          {/* bottom border line */}
          <View style={styles.tabBarBorder} />
        </View>

        {/* ── Content ── */}
        <ScrollView
          style={styles.contentScroll}
          contentContainerStyle={styles.contentContainer}
          showsVerticalScrollIndicator={false}
        >
          {/* Section heading */}
          <Text style={styles.sectionTitle}>{currentTab.sectionTitle}</Text>

          {/* Description paragraphs */}
          {currentTab.description.map((para, i) => (
            <Text key={i} style={styles.paragraph}>
              {para}
            </Text>
          ))}

          {/* Image cards */}
          {currentTab.items?.map((item, i) => (
            <PostureImageCard key={i} item={item} />
          ))}

          {/* Bottom spacer so last image isn't hidden behind button */}
          <View style={{ height: 24 }} />
        </ScrollView>

        {/* ── Close button ── */}
        <View style={styles.footer}>
          <TouchableOpacity style={styles.closeBtn} onPress={onClose} activeOpacity={0.85}>
            <Text style={styles.closeBtnText}>CLOSE</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </Modal>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const BLUE = '#2979FF';
const RED_LABEL = '#D32F2F';
const GREEN_LABEL = '#2E7D32';
const BORDER = '#E0E0E0';

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },

  // ── Header
  header: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#111111',
    letterSpacing: 0.3,
  },

  // ── Tab bar
  tabBarWrapper: {
    position: 'relative',
    paddingHorizontal: 12,
  },
  tabBar: {
    paddingHorizontal: 12,
    paddingBottom: 0,
    gap: 0,
  },
  tabRow: {
    flexDirection: 'row',
    //justifyContent: 'space-between', // evenly spread all 5 tabs
  },
  tabItem: {

      paddingHorizontal: 14,
      alignItems: 'center',
      paddingVertical: 10,
      borderBottomColor: 'transparent',
      borderBottomWidth: 3,

    /*
    paddingHorizontal: 10,
    alignItems: 'center',
    paddingVertical: 10,
    position: 'relative',
    borderBottomColor: 'transparent',
    marginBottom: 0,
    borderBottomWidth: 3,
    */
  },
  tabItemActive: {
    borderBottomColor: BLUE,
  },
  tabText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#888888',
  },
  tabTextActive: {
    color: BLUE,
    fontWeight: '700',
  },
  tabBarBorder: {
    height: 1,
    backgroundColor: BORDER,
    marginTop: -1,
  },

  // ── Content
  contentScroll: {
    flex: 1,
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111111',
    marginBottom: 12,
  },
  paragraph: {
    fontSize: 14,
    color: '#444444',
    lineHeight: 21,
    marginBottom: 10,
  },

  // ── Image cards
  imageCard: {
    marginTop: 12,
    borderRadius: 10,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: '#FAFAFA',
  },
  postureImage: {
    width: '100%',
    backgroundColor: '#EEEEEE',
  },
  imageLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  labelDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  dotGreen: {
    backgroundColor: '#43A047',
  },
  dotRed: {
    backgroundColor: '#E53935',
  },
  imageLabel: {
    fontSize: 13,
    fontWeight: '600',
  },
  labelGood: {
    color: GREEN_LABEL,
  },
  labelBad: {
    color: RED_LABEL,
  },

  // ── Footer / close button
  footer: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: BORDER,
    backgroundColor: '#FFFFFF',
  },
  closeBtn: {
    backgroundColor: BLUE,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
  },
  closeBtnText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
});

export default LearnPosture;