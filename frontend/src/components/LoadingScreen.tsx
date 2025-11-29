export default function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center z-50">
      <div className="text-center">
        <div className="relative inline-block">
          <div className="w-20 h-20 border-4 border-purple-200 border-t-purple-500 rounded-full animate-spin"></div>
          <div className="absolute inset-0 w-20 h-20 border-4 border-transparent border-b-pink-500 rounded-full animate-spin animation-delay-150"></div>
        </div>
        <p className="mt-6 text-white/80 font-medium">Loading ShikshaSetu...</p>
      </div>
    </div>
  );
}
