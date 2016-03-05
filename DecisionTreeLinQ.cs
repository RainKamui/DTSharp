using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Data;

namespace Algorithm {
	class Node {
		public int FeatureIndex { get; set; }
		public string Symbol { get; set; }
		public bool Leaf { get; set; }
		public List<Node> Childs { get; set; }
		public Node() {
			Childs = new List<Node>();
			Leaf = false;
		}
	}
	class BinaryNode {
		public int FeatureIndex { get; set; }
		public string Symbol { get; set; }
		public bool Leaf { get; set; }
		public int LeafCount { get; set; }
		public double R { get; set; }
		public double Alpha { get; set; }
		public BinaryNode Left { get; set; }
		public BinaryNode Right { get; set; }
		public BinaryNode() {

		}
		public BinaryNode(string symbol, int featureIndex, bool leaf) {
			this.FeatureIndex = featureIndex;
			this.Symbol = symbol;
			this.R = 1.0d;
			this.Leaf = leaf;
			this.Left = null;
			this.Right = null;
		}
	}
	class DecisionTreeLinQ {
		public DataSet _dataSet = new DataSet();

		public int ClassifyIndex { get; set; }
		public int DefaultTable { get; set; }

		private int _recordCount = 0;
		private int _featureCount = 0;

		#region ID3ANDC4.5
		private double __gda(IEnumerable<DataRow> data, int featureIndex, bool gdar = false) {
			var q = 0.0d;
			data.GroupBy(r => r.Field<string>(ClassifyIndex)).Select(r => r.Count()).ToList().ForEach(c => {
				var tmp = (double)c / (double)data.Count(); q += tmp * _log(tmp);
			});
			q *= -1;

			var ce = 0.0d;
			data.GroupBy(r => r.Field<string>(featureIndex)).ToList().ForEach(e => {
				var qa = 0.0d;
				e.GroupBy(c => c.Field<string>(ClassifyIndex)).ToList().ForEach(a => {
					var tmp = (double)a.Count() / e.Count(); qa += tmp * _log(tmp);
				});
				qa *= -1;
				ce += ((double)e.Count() / (double)data.Count()) * qa;
			});

			var gda = q - ce;
			return !gdar ? gda : gda / q;
		}
		/// <summary>
		///  ID3 C4.5算法实现
		/// </summary>
		/// <param name="data">数据集</param>
		/// <param name="attributes">需要查询的特征</param>
		/// <returns></returns>
		private Node ___common(IEnumerable<DataRow> data, List<int> attributes, bool id3 = true, double epsilon = -1) {
			var node = new Node();
			var maxIndex = -1;
			var maxGain = 0.0d;
			//找出当前数据集增益最大的特征
			foreach (var i in attributes) {
				var gain = __gda(data, i, !id3);
				if (gain > maxGain) {
					maxGain = gain;
					maxIndex = i;
				}
			}

			node.FeatureIndex = maxIndex;
			node.Symbol = "Feature_" + maxIndex;

			if (epsilon != -1 && maxGain < epsilon) {
				//如果最大信息增益小于阀值，则选择实例数最多的类作为该节点的类标记
				var q = data.GroupBy(r => r.Field<string>(ClassifyIndex)).OrderByDescending(d => d.Count()).Select(r => r.Key);
				var query =
					from d in data
					group d by d[ClassifyIndex] into g
					orderby g.Count() descending
					select new { g.Key };
				node.FeatureIndex = maxIndex;
				node.Symbol = q.First().ToString();
				node.Leaf = true;
				return node;
			}

			//找出最大增益特征的所有取值，存入splitAttribute
			var splitAttribute = new List<string>();
			data.GroupBy(r => r.Field<string>(maxIndex)).ToList().ForEach(c => splitAttribute.Add(c.Key));
			//遍历取值
			foreach (var attr in splitAttribute) {
				//找出最大增益特征的一个划分
				var ds = data.Where(r => r.Field<string>(maxIndex) == attr);
				//对该划分的分类GroupBy
				var c = ds.GroupBy(r => r.Field<string>(ClassifyIndex));
				Node childNode = null;
				//如果该划分中只有一个分类
				if (c.Count() == 1) {
					//设置这个分类的Symbol为该划分的值，并设置叶子节点属性为真
					childNode = new Node();
					childNode.FeatureIndex = maxIndex;
					childNode.Symbol = c.Select(r => r.Key).First().ToString();
					childNode.Leaf = true;
				}
				else {
					//新建一个查询特征列表并在其中删除当前最大增益特征
					var newAttributes = new List<int>();
					attributes.ForEach(e => { if (e != maxIndex) newAttributes.Add(e); });
					//在该划分中递归查找下一个最大增益特征
					childNode = ___common(ds, newAttributes);
				}
				node.Childs.Add(childNode);
			}
			return node;
		}

		public Node _id3(double epsilon = -1) {
			var data = _dataSet.Tables[DefaultTable].AsEnumerable();
			var list = new List<int>();
			for (int i = 1; i < _featureCount; i++) {
				if (i != ClassifyIndex)
					list.Add(i);
			}
			return ___common(data, list, true, epsilon);
		}

		public Node _c45(double epsilon = -1) {
			var data = _dataSet.Tables[DefaultTable].AsEnumerable();
			var list = new List<int>();
			for (int i = 1; i < _featureCount; i++) {
				if (i != ClassifyIndex)
					list.Add(i);
			}
			return ___common(data, list, false, epsilon);
		}
		#endregion

		#region CART
		private double _gini(IEnumerable<DataRow> data) {
			var sum = 0.0d;
			data.GroupBy(r => r.Field<string>(ClassifyIndex)).ToList().ForEach(e => {
				var p = ((double)e.Count() / (double)data.Count());
				sum += Math.Pow(p, 2);
			});
			return 1 - sum;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="data"></param>
		/// <param name="featureIndex">_1：最优特征值，_2：最优切分点， _3：gini指数, _4：该划分点的两个划分集</param>
		/// <returns></returns>
		private Tuple<int, string, double, Tuple<IEnumerable<DataRow>, IEnumerable<DataRow>>> _giniCondition(IEnumerable<DataRow> data, int featureIndex) {
			var minValue = string.Empty;
			var minGini = 2d;
			//存储最优切分点的两个划分集，_1为是该切分点，_2为不是该切分点
			Tuple<IEnumerable<DataRow>, IEnumerable<DataRow>> optimalPoint = null;
			data.GroupBy(r => r.Field<string>(featureIndex)).ToList().ForEach(e => {
				var dy = data.Where(r => r.Field<string>(featureIndex) == e.Key);
				var dn = data.Where(r => r.Field<string>(featureIndex) != e.Key);
				var giniCon = ((double)dy.Count() / (double)data.Count()) * _gini(dy) + ((double)dn.Count() / (double)data.Count()) * _gini(dn);
				if (giniCon < minGini) {
					minGini = giniCon;
					minValue = e.Key;
					optimalPoint = new Tuple<IEnumerable<DataRow>, IEnumerable<DataRow>>(dy, dn);
				}
			});
			return new Tuple<int, string, double, Tuple<IEnumerable<DataRow>, IEnumerable<DataRow>>>(featureIndex, minValue, minGini, optimalPoint);
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="data">DataSet</param>
		/// <param name="beforeCount">划分之前的集合数据数量</param>
		/// <param name="attributes">当前划分的可用attributes</param>
		/// <param name="featureIndex"></param>
		/// <param name="epsilon"></param>
		/// <param name="minClassify"></param>
		/// <returns></returns>
		private BinaryNode __cart(IEnumerable<DataRow> data, List<int> attributes, double epsilon = -1, int minClassify = 1, int featureIndex = -1) {
			BinaryNode node = null;

			//先对当前数据集的分类分组
			var classify = data.GroupBy(r => r.Field<string>(ClassifyIndex));
			if (classify.Count() <= minClassify) {
				node = new BinaryNode(classify.Select(e => e.Key).First().ToString(), featureIndex, true);
				//叶子节点代价R(t) = dataCount / recordCount
				node.R = (double)data.Count() / (double)_recordCount;
			}
			//对当前数据集求gini指数，若是小于阀值，则取样本最多的分类数
			else if (epsilon != -1 && _gini(data) < epsilon) {
				var maxCountClassify = classify.OrderByDescending(e => e.Count()).First();
				node = new BinaryNode(maxCountClassify.Key, featureIndex, true);
				node.R = (double)data.Count() / (double)_recordCount;
			}
			else {
				//不能分为叶节点，对该节点进行递归寻找最小gini划分并递归
				Tuple<int, string, double, Tuple<IEnumerable<DataRow>, IEnumerable<DataRow>>> minGini = null;
				foreach (var attr in attributes) {
					var tmp = _giniCondition(data, attr);
					minGini = minGini == null ? tmp : tmp.Item1 < minGini.Item1 ? tmp : minGini;
				}
				var newAttribute = new List<int>();
				attributes.ForEach(e => {
					if (e != minGini.Item1) newAttribute.Add(e);
				});
				node = new BinaryNode("Is feature_" + minGini.Item1 + " - " + minGini.Item2 + "?", minGini.Item1, false);
				//节点代价R(t) = r(t) * p(t)
				//r(t) 是节点的误差率 r(t) = |划分点| / |该节点数据|
				//p(t) 是节点上的数据所占总数据比例 p(t) = |该节点数据| / |总数据|
				//所以 R(t) = |划分点| / |总数据|
				var Rt = (double)minGini.Item4.Item1.Count() /  (double)_recordCount;

				node.R = Rt;
				node.Left = __cart(minGini.Item4.Item1, newAttribute, epsilon, minClassify, minGini.Item1);
				node.Right = __cart(minGini.Item4.Item2, newAttribute, epsilon, minClassify, minGini.Item1);
			}

			return node;
		}
		private void __setAlpha(BinaryNode node) {
			if (node != null) {
				if (node.Leaf == false) {
					var RTt = __computeRTt(node);
					var leafCount = __computeLeafCount(node);
					node.LeafCount = leafCount;
					var alpha = (node.R / (double)RTt) / (double)(leafCount - 1);
					node.Alpha = alpha;
				}
				__setAlpha(node.Left);
				__setAlpha(node.Right);
			}
		}
		private double __computeRTt(BinaryNode node) {
			//return node == null ? 0d : node.Leaf == true ? node.R :__computeAlpha(node.Left) + __computeAlpha(node.Right);
			if (node == null)
				return 0d;
			else if (node.Leaf == true)
				return node.R;
			else
				return __computeRTt(node.Left) + __computeRTt(node.Right);
		}

		private int __computeLeafCount(BinaryNode node) {
			if (node != null) {
				var c = node.Leaf == true ? 1 : 0;
				return __computeLeafCount(node.Left) + __computeLeafCount(node.Right) + c;
			}
			return 0;
		}
		private void _ccp(BinaryNode node) {
			//设置所有节点的alpha
			__setAlpha(node);

			var minAlpha = Double.MaxValue;
			BinaryNode minNode = null;
			var queue = new Queue<BinaryNode>();
			queue.Enqueue(node);
			while (queue.Count > 0) {
				var tmp = queue.Dequeue();
				if (tmp.Left != null) queue.Enqueue(tmp.Left);
				if (tmp.Right != null) queue.Enqueue(tmp.Right);
				
				if (tmp.Alpha < minAlpha) {
					minAlpha = tmp.Alpha;
					minNode = tmp;
				}
				else if (tmp.Alpha == minAlpha && tmp.LeafCount > minNode.LeafCount) {
					minNode = tmp;
				}
			}

			//剪枝
			minNode.Left = null;
			minNode.Right = null;
		}
		public BinaryNode _cart(bool cut = false) {
			var table = _dataSet.Tables[DefaultTable].AsEnumerable();
			var list = new List<int>();
			for (int i = 1; i < _featureCount; i++) {
				if (i != ClassifyIndex) {
					list.Add(i);
				}
			}
			var tree = __cart(table, list);
			if (cut) _ccp(tree);
			return tree;
		}
		#endregion
		#region Helper
		public void LoadText(string text, string split) {
			var lines = Regex.Split(text, Environment.NewLine);
			var table = new DataTable();
			_featureCount = Regex.Split(lines[0], split).Count();
			for (int i = 0; i < _featureCount; i++) {
				table.Columns.Add("feature_" + i, System.Type.GetType("System.String"));
			}

			for (int i = 0; i < lines.Length; i++) {
				var elements = Regex.Split(lines[i], split);
				var row = table.NewRow();
				for (int j = 0; j < _featureCount; j++) {
					row[j] = elements[j];
				}
				table.Rows.Add(row);
			}
			_recordCount = lines.Count();
			_dataSet.Tables.Add(table);
		}


		private void _printTree(Node node) {
			if (node.Leaf) {
				Console.WriteLine("Leaf Node : " + node.FeatureIndex + " " + node.Symbol);
			}
			else {
				Console.WriteLine("Classify Node : " + node.Symbol);
				node.Childs.ForEach(n => _printTree(n));
			}
		}
		private void _printTree(BinaryNode node) {
			if (node.Leaf) {
				Console.WriteLine("Leaf Node : " + node.FeatureIndex + " " + node.Symbol);
			}
			else {
				Console.WriteLine("Classify Node : " + node.Symbol);
					_printTree(node.Left);
					_printTree(node.Right);
			}
		}
		private static double _log(double num) {
			return Math.Log(num, 2);
		}
		#endregion
	}
}
