export default function PageHeader({ eyebrow, title, subtitle }) {
  return (
    <div className="page-banner">
      <div className="page-banner-inner">
        <div className="page-eyebrow">{eyebrow}</div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-sub">{subtitle}</p>}
      </div>
    </div>
  );
}
