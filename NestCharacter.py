# -*- coding: utf-8 -*-
# coding:utf-8
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import argparse


def create_text_fill_art(
        large_text: str,
        small_text: str,
        large_font_size: int,
        small_font_size: int,
        font_path: str,
        output_filename: str,
        step_x: int,
        step_y: int,
        wrap_after: int,
        background_color: str = "white",
        text_color: str = "black"
):
    print("开始生成图像，参数如下:")
    print(f"  - 大字文本: {large_text[:30]}...")
    print(f"  - 换行设置: 每 {wrap_after} 字换行" if wrap_after > 0 else "  - 换行设置: 不换行")
    print(f"  - 小字文本: {small_text}")
    print(f"  - 大字字号: {large_font_size}")
    print(f"  - 小字字号: {small_font_size}")

    final_step_x = step_x if step_x is not None else max(2, int(small_font_size * 0.85))
    final_step_y = step_y if step_y is not None else max(2, int(small_font_size * 0.85))
    print(f"  - 小字水平间距: {final_step_x}px")
    print(f"  - 小字垂直间距: {final_step_y}px")
    print("-" * 30)

    start_time = time.time()
    if wrap_after > 0:
        lines = [large_text[i:i + wrap_after] for i in range(0, len(large_text), wrap_after)]
    else:
        lines = [large_text]

    LINE_SPACING_FACTOR = 1.2
    line_height = int(large_font_size * LINE_SPACING_FACTOR)

    padding_y = int(large_font_size * 0.6)

    text_block_height = line_height * (len(lines) - 1) + large_font_size
    IMAGE_HEIGHT = text_block_height + 2 * padding_y

    num_chars_for_width = wrap_after if wrap_after > 0 else len(large_text)
    IMAGE_WIDTH = int(num_chars_for_width * large_font_size * 1.05)

    try:
        large_font = ImageFont.truetype(font_path, large_font_size)
        small_font = ImageFont.truetype(font_path, small_font_size)
    except IOError:
        print(f"错误：无法在 '{font_path}' 找到字体文件。")
        return

    final_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), background_color)
    draw_final = ImageDraw.Draw(final_image)

    small_text_index = 0

    print("开始逐字生成模板并填充...")

    current_y = padding_y
    for line in lines:
        try:
            # getlength is more accurate for multi-character strings than summing getbbox widths
            line_width = large_font.getlength(line)
        except AttributeError:
            # Fallback for older Pillow versions
            line_width = draw_final.textlength(line, font=large_font)

        line_start_x = (IMAGE_WIDTH - line_width) / 2
        current_x = line_start_x

        # 按字符遍历
        for char in line:
            print(f"  - 正在处理大字: '{char}'")

            char_mask = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
            draw_char_mask = ImageDraw.Draw(char_mask)
            draw_char_mask.text((current_x, current_y), char, font=large_font, fill=255)
            char_mask_array = np.array(char_mask)
            for y in range(0, IMAGE_HEIGHT, final_step_y):
                for x in range(0, IMAGE_WIDTH, final_step_x):
                    # 检查坐标是否在单字模板的笔画内
                    if char_mask_array[y, x] > 128:
                        char_to_draw = small_text[small_text_index % len(small_text)]
                        draw_final.text((x, y), char_to_draw, font=small_font, fill=text_color)
                        small_text_index += 1

            try:
                char_width = large_font.getlength(char)
            except AttributeError:
                char_width = draw_final.textlength(char, font=large_font)
            current_x += char_width
        current_y += line_height

    print("填充完成。正在保存图像...")

    # 保存最终结果
    final_image.save(output_filename)
    end_time = time.time()
    print("-" * 30)
    print(f"图像已保存为: {output_filename}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="一个命令行工具，用指定的小字填充大字的笔画来创造图像。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-lt', '--large-text', type=str,
                        default="夜饮东坡醒复醉，归来仿佛三更。家童鼻息已雷鸣。敲门都不应，倚杖听江声。长恨此身非我有，何时忘却营营。夜阑风静縠纹平。小舟从此逝，江海寄余生。",
                        help="被填充的大字文本。")
    parser.add_argument('-st', '--small-text', type=str,
                        default="中国社会各阶级的分析（一九二五年十二月一日）谁是我们的敌人？谁是我们的朋友？这个问题是革命的首要问题。中国过去一切革命斗争成效甚少，其基本原因就是因为不能团结真正的朋友，以攻击真正的敌人。革命党是群众的向导，在革命中未有革命党领错了路而革命不失败的。我们的革命要有不领错路和一定成功的把握，不可不注意团结我们的真正的朋友，以攻击我们的真正的敌人。我们要分辨真正的敌友，不可不将中国社会各阶级的经济地位及其对于革命的态度，作一个大概的分析。中国社会各阶级的情况是怎样的呢？地主阶级和买办阶级。在经济落后的半殖民地的中国，地主阶级和买办阶级完全是国际资产阶级的附庸，其生存和发展，是附属于帝国主义的。这些阶级代表中国最落后的和最反动的生产关系，阻碍中国生产力的发展。他们和中国革命的目的完全不兼容。特别是大地主阶级和大买办阶级，他们始终站在帝国主义一边，是极端的反革命派。其政治代表是国家主义派和国民党右派。中产阶级。这个阶级代表中国城乡资本主义的生产关系。中产阶级主要是指民族资产阶级，他们对于中国革命具有矛盾的态度：他们在受外资打击、军阀压迫感觉痛苦时，需要革命，赞成反帝国主义反军阀的革命运动；但是当着革命在国内有本国无产阶级的勇猛参加，在国外有国际无产阶级的积极援助，对于其欲达到大资产阶级地位的阶级的发展感觉到威胁时，他们又怀疑革命。其政治主张为实现民族资产阶级一阶级统治的国家。有一个自称为戴季陶“真实信徒”的，在北京《晨报》上发表议论说：“举起你的左手打倒帝国主义，举起你的右手打倒共产党。”这两句话，画出了这个阶级的矛盾惶遽状态。他们反对以阶级斗争学说解释国民党的民生主义，他们反对国民党联俄和容纳共产党及左派分子。但是这个阶级的企图——实现民族资产阶级统治的国家，是完全行不通的，因为现在世界上的局面，是革命和反革命两大势力作最后斗争的局面。这两大势力竖起了两面大旗：一面是红色的革命的大旗，第三国际高举着，号召全世界一切被压迫阶级集合于其旗帜之下；一面是白色的反革命的大旗，国际联盟高举着，号召全世界一切反革命分子集合于其旗帜之下。那些中间阶级，必定很快地分化，或者向左跑入革命派，或者向右跑入反革命派，没有他们“独立”的余地。所以，中国的中产阶级，以其本阶级为主体的“独立”革命思想，仅仅是一个幻想。小资产阶级。如自耕农，手工业主，小知识阶层——学生界、中小学教员、小员司、小事务员、小律师，小商人等都属于这一类。这一个阶级，在人数上，在阶级性上，都值得大大注意。自耕农和手工业主所经营的，都是小生产的经济。这个小资产阶级内的各阶层虽然同处在小资产阶级经济地位，但有三个不同的部分。第一部分是有余钱剩米的，即用其体力或脑力劳动所得，除自给外，每年有余剩。这种人发财观念极重，对赵公元帅礼拜最勤，虽不妄想发大财，却总想爬上中产阶级地位。他们看见那些受人尊敬的小财东，往往垂着一尺长的涎水。这种人胆子小，他们怕官，也有点怕革命。因为他们的经济地位和中产阶级颇接近，故对于中产阶级的宣传颇相信，对于革命取怀疑的态度。这一部分人在小资产阶级中占少数，是小资产阶级的右翼。第二部分是在经济上大体上可以自给的。这一部分人比较第一部分人大不相同，他们也想发财，但是赵公元帅总不让他们发财，而且因为近年以来帝国主义、军阀、封建地主、买办大资产阶级的压迫和剥削，他们感觉现在的世界已经不是从前的世界。他们觉得现在如果只使用和从前相等的劳动，就会不能维持生活。必须增加劳动时间，每天起早散晚，对于职业加倍注意，方能维持生活。他们有点骂人了，骂洋人叫“洋鬼子”，骂军阀叫“抢钱司令”，骂土豪劣绅叫“为富不仁”。对于反帝国主义反军阀的运动，仅怀疑其未必成功（理由是：洋人和军阀的来头那么大），不肯贸然参加，取了中立的态度，但是绝不反对革命。这一部分人数甚多，大概占小资产阶级的一半。第三部分是生活下降的。这一部分人好些大概原先是所谓殷实人家，渐渐变得仅仅可以保住，渐渐变得生活下降了。他们每逢年终结账一次，就吃惊一次，说：“咳，又亏了！”这种人因为他们过去过着好日子，后来逐年下降，负债渐多，渐次过着凄凉的日子，“瞻念前途，不寒而栗”。这种人在精神上感觉的痛苦很大，因为他们有一个从前和现在相反的比较。这种人在革命运动中颇要紧，是一个数量不小的群众，是小资产阶级的左翼。以上所说小资产阶级的三部分，对于革命的态度，在平时各不相同；但到战时，即到革命潮流高涨、可以看得见胜利的曙光时，不但小资产阶级的左派参加革命，中派亦可参加革命，即右派分子受了无产阶级和小资产阶级左派的革命大潮所裹挟，也只得附和着革命。我们从一九二五年的五卅运动和各地农民运动的经验看来，这个断定是不错的。半无产阶级。此处所谓半无产阶级，包含：（一）绝大部分半自耕农，（二）贫农，（三）小手工业者，（四）店员，（五）小贩等五种。绝大部分半自耕农和贫农是农村中一个数量极大的群众。所谓农民问题，主要就是他们的问题。半自耕农、贫农和小手工业者所经营的，都是更细小的小生产的经济。绝大部分半自耕农和贫农虽同属半无产阶级，但其经济状况仍有上、中、下三个细别。半自耕农，其生活苦于自耕农，因其食粮每年大约有一半不够，须租别人田地，或者出卖一部分劳动力，或经营小商，以资弥补。春夏之间，青黄不接，高利向别人借债，重价向别人籴粮，较之自耕农的无求于人，自然景遇要苦，但是优于贫农。因为贫农无土地，每年耕种只得收获之一半或不足一半；半自耕农则租于别人的部分虽只收获一半或不足一半，然自有的部分却可全得。故半自耕农的革命性优于自耕农而不及贫农。贫农是农村中的佃农，受地主的剥削。其经济地位又分两部分。一部分贫农有比较充足的农具和相当数量的资金。此种农民，每年劳动结果，自己可得一半。不足部分，可以种杂粮、捞鱼虾、饲鸡豕，或出卖一部分劳动力，勉强维持生活，于艰难竭蹶之中，存聊以卒岁之想。故其生活苦于半自耕农，然较另一部分贫农为优。其革命性，则优于半自耕农而不及另一部分贫农。所谓另一部分贫农，则既无充足的农具，又无资金，肥料不足，土地歉收，送租之外，所得无几，更需要出卖一部分劳动力。荒时暴月，向亲友乞哀告怜，借得几斗几升，敷衍三日五日，债务丛集，如牛负重。他们是农民中极艰苦者，极易接受革命的宣传。小手工业者所以称为半无产阶级，是因为他们虽然自有简单的生产手段，且系一种自由职业，但他们也常常被迫出卖一部分劳动力，其经济地位略与农村中的贫农相当。因其家庭负担之重，工资和生活费用之不相称，时有贫困的压迫和失业的恐慌，和贫农亦大致相同。店员是商店的雇员，以微薄的薪资，供家庭的费用，物价年年增长，薪给往往须数年一增，偶与此辈倾谈，便见叫苦不迭。其地位和贫农及小手工业者不相上下，对于革命宣传极易接受。小贩不论肩挑叫卖，或街畔摊售，总之本小利微，吃着不够。其地位和贫农不相上下，其需要一个变更现状的革命，也和贫农相同。无产阶级。现代工业无产阶级约二百万人。中国因经济落后，故现代工业无产阶级人数不多。二百万左右的产业工人中，主要为铁路、矿山、海运、纺织、造船五种产业的工人，而其中很大一个数量是在外资产业的奴役下。工业无产阶级人数虽不多，却是中国新的生产力的代表者，是近代中国最进步的阶级，做了革命运动的领导力量。我们看四年以来的罢工运动，如海员罢工、铁路罢工、开滦和焦作煤矿罢工、沙面罢工以及“五卅”后上海香港两处的大罢工所表现的力量，就可知工业无产阶级在中国革命中所处地位的重要。他们所以能如此，第一个原因是集中。无论哪种人都不如他们的集中。第二个原因是经济地位低下。他们失了生产手段，剩下两手，绝了发财的望，又受着帝国主义、军阀、资产阶级的极残酷的待遇，所以他们特别能战斗。都市苦力工人的力量也很可注意。以码头搬运夫和人力车夫占多数，粪夫清道夫等亦属于这一类。他们除双手外，别无长物，其经济地位和产业工人相似，惟不及产业工人的集中和在生产上的重要。中国尚少新式的资本主义的农业。所谓农村无产阶级，是指长工、月工、零工等雇农而言。此等雇农不仅无土地，无农具，又无丝毫资金，只得营工度日。其劳动时间之长，工资之少，待遇之薄，职业之不安定，超过其它工人。此种人在乡村中是最感困难者，在农民运动中和贫农处于同一紧要的地位。此外，还有数量不小的游民无产者，为失了土地的农民和失了工作机会的手工业工人。他们是人类生活中最不安定者。他们在各地都有秘密组织，如闽粤的“三合会”，湘鄂黔蜀的“哥老会”，皖豫鲁等省的“大刀会”，直隶及东三省的“在理会”，上海等处的“青帮”，都曾经是他们的政治和经济斗争的互助团体。处置这一批人，是中国的困难的问题之一。这一批人很能勇敢奋斗，但有破坏性，如引导得法，可以变成一种革命力量。综上所述，可知一切勾结帝国主义的军阀、官僚、买办阶级、大地主阶级以及附属于他们的一部分反动知识界，是我们的敌人。工业无产阶级是我们革命的领导力量。一切半无产阶级、小资产阶级，是我们最接近的朋友。那动摇不定的中产阶级，其右翼可能是我们的敌人，其左翼可能是我们的朋友——但我们要时常提防他们，不要让他们扰乱了我们的阵线。",
                        help="用于填充的小字文本。")
    parser.add_argument('-lfs', '--large-font-size', type=int, default=400,
                        help="被填充的大字字号。\n默认: 200")
    parser.add_argument('-sfs', '--small-font-size', type=int, default=12,
                        help="用于填充的小字字号。\n默认: 12")
    parser.add_argument('-w', '--wrap-after', type=int, default=10,
                        help="设置大字文本每行显示的字数。\n设置为 0 表示不自动换行。\n默认: 0")
    parser.add_argument('--step-x', type=int, default=13,
                        help="小字填充的水平步长/间距（像素）。\n默认: 根据小字号自动计算")
    parser.add_argument('--step-y', type=int, default=13,
                        help="小字填充的垂直步长/行间距（像素）。\n默认: 根据小字号自动计算")
    parser.add_argument('-fp', '--font-path', type=str, default='C:\Windows\Fonts\msyh.ttc',
                        help="中文字體文件的路徑。\nmacOS 推薦使用 Hiragino Sans GB: '/System/Library/Fonts/Hiragino Sans GB.ttc'")
    parser.add_argument('-o', '--output', type=str, default="output_art.png",
                        help="输出图像的文件名。\n默认: 'output_art.png'")

    args = parser.parse_args()

    create_text_fill_art(
        large_text=args.large_text,
        small_text=args.small_text,
        large_font_size=args.large_font_size,
        small_font_size=args.small_font_size,
        font_path=args.font_path,
        output_filename=args.output,
        step_x=args.step_x,
        step_y=args.step_y,
        wrap_after=args.wrap_after
    )